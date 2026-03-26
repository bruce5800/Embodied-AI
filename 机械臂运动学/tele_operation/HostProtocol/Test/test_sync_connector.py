import unittest
import sys
import os
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SDK.sync_connector import SyncConnector
from SDK.port_handler import PortHandler
from SDK.global_state import Address, Result, ErrorCode


class TestSyncConnectorWrite(unittest.TestCase):
    def setUp(self):
        self.mock_port = MagicMock(spec=PortHandler)
        self.connector = SyncConnector(self.mock_port)
    
    def test_write_single_address(self):
        """测试单地址写入"""
        addr = Address.TARGET_POSITION
        self.mock_port.write_port.return_value = None
        
        result = self.connector.write(1, addr, 1234)
        
        self.assertTrue(result.is_success())
        self.mock_port.write_port.assert_called_once()
    
    def test_write_multiple_addresses(self):
        """测试多地址写入"""
        addrs = [Address.TARGET_POSITION, Address.MAX_POSITION]
        values = [1234, 5678]
        self.mock_port.write_port.return_value = None
        
        result = self.connector.write(1, addrs, values)
        
        self.assertTrue(result.is_success())
        self.mock_port.write_port.assert_called_once()
    
    def test_write_invalid_id(self):
        """测试无效ID"""
        with self.assertRaises(ValueError):
            self.connector.write(121, Address(0x1000, 4), 1234)
    
    def test_write_error_handling(self):
        """测试写入错误处理"""
        self.mock_port.write_port.side_effect = Exception("Write error")
        
        result = self.connector.write(1, Address(0x1000, 4), 1234)
        
        self.assertFalse(result.is_success())
        self.assertEqual(result.error, ErrorCode.WRITE_ERROR)


if __name__ == '__main__':
    unittest.main()