"""
交互式打字对话：DeepSeek + MCP 工具自动调用

依赖安装：
  pip install openai mcp -i https://pypi.tuna.tsinghua.edu.cn/simple

环境变量：
  在 PowerShell 当前会话设置：
    $env:DEEPSEEK_API_KEY = "你的DeepSeek API Key"

MCP服务器：
  确保已启动本地 MCP SSE 服务（默认 http://localhost:8000/sse）
  例如：python mcp_server.py
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional

from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPDeepSeekChat:
    def __init__(self, api_key: str, mcp_server_url: str = "http://localhost:8000/sse"):
        self.deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.mcp_server_url = mcp_server_url
        self.available_tools: List[Any] = []

    async def get_mcp_tools(self) -> List[Any]:
        async with sse_client(url=self.mcp_server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.available_tools = tools.tools
                return self.available_tools

    def format_tools_for_deepseek(self) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for tool in self.available_tools:
            entry = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"调用{tool.name}工具",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
            }
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                entry["function"]["parameters"] = tool.inputSchema
            formatted.append(entry)
        return formatted

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        async with sse_client(url=self.mcp_server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result.content[0].text if result.content else None

    async def chat_once(self, messages: List[Dict[str, Any]], tools_payload: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        response = self.deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools_payload if tools_payload else None,
            tool_choice="auto" if tools_payload else None,
            stream=False,
        )
        return response.choices[0].message

    async def interactive_loop(self):
        # 初始化工具（可失败时降级为无工具对话）
        tools_payload: Optional[List[Dict[str, Any]]] = None
        try:
            await self.get_mcp_tools()
            tools_payload = self.format_tools_for_deepseek()
            if tools_payload:
                print(f"已加载 {len(tools_payload)} 个 MCP 工具。AI 将自动选择是否调用。")
            else:
                print("MCP 未提供工具，进行纯聊天模式。")
        except Exception as e:
            print(f"连接 MCP 服务器失败，进行纯聊天模式：{e}")

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": "你叫小明，你是个很厉害的我的小帮手，你既能跟我聊天，又能帮我干活，还能报我操作机械臂，1号电机是腰，2号电机是大臂，3号电机是小臂，4，5号电机是腕关节，6号电机是夹爪",
            }
        ]

        print("\n=== 交互式聊天开始（输入 exit 或 q 退出）===")
        while True:
            try:
                user_input = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break

            if user_input.lower() in {"exit", "quit", "q"}:
                print("再见！")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            # 可能出现多轮工具调用，直到返回无工具调用的最终回复
            while True:
                assistant_message = await self.chat_once(messages, tools_payload)

                # 先把助手消息（可能包含 tool_calls）加入上下文
                assistant_record = {"role": "assistant", "content": assistant_message.content}
                if getattr(assistant_message, "tool_calls", None):
                    assistant_record["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ]
                messages.append(assistant_record)

                # 如需调用工具，则执行并把结果作为 tool 角色消息加入
                tool_calls = getattr(assistant_message, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.function.name
                        arg_text = tc.function.arguments or "{}"
                        try:
                            args = json.loads(arg_text)
                        except Exception:
                            args = {}
                        try:
                            result_text = await self.call_mcp_tool(tool_name, args)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(result_text),
                            })
                        except Exception as e:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"工具调用失败: {str(e)}",
                            })
                    # 执行完工具后继续循环，让模型基于工具结果给出最终回复
                    continue

                # 无工具调用，打印最终回复并结束本轮
                final_text = assistant_message.content or "(无内容)"
                print(f"助手: {final_text}")
                break


async def main():
    api_key = "sk-6bc37dec91f84ed19278eb9c2ed9cd40"
    if not api_key:
        print("未检测到环境变量 DEEPSEEK_API_KEY。请设置后重试。")
        return

    chat = MCPDeepSeekChat(api_key=api_key)
    await chat.interactive_loop()


if __name__ == "__main__":
    asyncio.run(main())

