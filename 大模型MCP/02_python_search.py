#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_python_search.py

在 Windows 的 D:\ 盘按文件名搜索文件的示例脚本。

用法示例：
  - 按通配符搜索：
    python 02_python_search.py "*.pdf"
  - 精确匹配并找到第一个后退出：
    python 02_python_search.py "readme.txt" --exact --first
  - 指定起始目录并排除部分目录：
    python 02_python_search.py "log" --start "D:\\projects" --exclude node_modules .git

说明：
  - 默认不区分大小写，可用 --case-sensitive 开启区分大小写。
  - 默认采用“包含或通配符”匹配：
      * 如果 pattern 中包含 * 或 ?，使用通配符匹配；
      * 否则使用“包含”匹配（文件名中包含即算匹配）。
  - 可使用 --exact 开启精确匹配（文件名完全相同）。
"""

import os
import argparse
import fnmatch
import sys
from typing import Iterator, List


def iter_files(start_dir: str, exclude_dirs: List[str]) -> Iterator[str]:
    """遍历起始目录下的所有文件，支持排除部分目录（按目录名匹配）。"""
    exclude_set = {d.lower() for d in (exclude_dirs or [])}
    for root, dirs, files in os.walk(start_dir, topdown=True):
        # 预先过滤需要排除的目录（忽略大小写）
        dirs[:] = [d for d in dirs if d.lower() not in exclude_set]
        for f in files:
            yield os.path.join(root, f)


def match_filename(path: str, pattern: str, exact: bool, case_insensitive: bool) -> bool:
    """对给定文件路径的文件名进行匹配。"""
    base = os.path.basename(path)
    if case_insensitive:
        base_cmp = base.lower()
        pattern_cmp = pattern.lower()
    else:
        base_cmp = base
        pattern_cmp = pattern

    if exact:
        return base_cmp == pattern_cmp

    # 非精确匹配：如果 pattern 包含通配符则使用 fnmatch，否则按“包含”匹配
    has_wildcard = any(ch in pattern_cmp for ch in "*?")
    if has_wildcard:
        return fnmatch.fnmatch(base_cmp, pattern_cmp)
    return pattern_cmp in base_cmp


def search(
    start_dir: str,
    pattern: str,
    exact: bool = False,
    case_insensitive: bool = True,
    first: bool = False,
    exclude_dirs: List[str] | None = None,
) -> List[str]:
    """在 start_dir 下按给定匹配规则搜索文件名，返回匹配到的完整路径列表。"""
    results: List[str] = []
    try:
        for path in iter_files(start_dir, exclude_dirs or []):
            try:
                if match_filename(path, pattern, exact, case_insensitive):
                    results.append(path)
                    if first:
                        break
            except Exception:
                # 某些特殊文件可能触发异常，直接跳过
                continue
    except (PermissionError, FileNotFoundError) as e:
        print(f"遍历目录时出现错误：{e}", file=sys.stderr)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在 D:\\ 盘按文件名搜索文件的示例脚本",
    )
    parser.add_argument(
        "pattern",
        help="要搜索的文件名或通配符模式，例如 'report.pdf' 或 '*.txt'",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="使用精确匹配（文件名完全相同）",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="区分大小写匹配（默认不区分）",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="找到第一个匹配后立即退出",
    )
    parser.add_argument(
        "--start",
        default="D:\\",
        help="起始搜索目录，默认 D:\\",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="排除的目录名列表（仅目录名，不含路径），默认排除系统目录",
    )

    args = parser.parse_args()

    start_dir = args.start
    if not os.path.exists(start_dir):
        print(f"起始目录不存在：{start_dir}", file=sys.stderr)
        sys.exit(1)

    # 默认排除部分系统目录，避免权限问题或冗余结果
    default_exclude = ["System Volume Information", "$Recycle.Bin"]
    exclude_dirs = args.exclude if args.exclude is not None else default_exclude

    results = search(
        start_dir=start_dir,
        pattern=args.pattern,
        exact=args.exact,
        case_insensitive=not args.case_sensitive,
        first=args.first,
        exclude_dirs=exclude_dirs,
    )

    if results:
        print(f"共找到 {len(results)} 个匹配：")
        for p in results:
            print(p)
    else:
        print("未找到匹配的文件。")


if __name__ == "__main__":
    main()