import time
from ctypes import *
import config


class PCA9685Driver:
    def __init__(self, dll_path="CH347DLLA64.DLL"):
        self.ch347 = windll.LoadLibrary(dll_path)
        self.dev_idx = 0
        self.i2c_addr = 0x40

        # 打开设备（全局唯一句柄）
        if self.ch347.CH347OpenDevice(self.dev_idx) == -1:
            raise RuntimeError("CH347 连接失败")

        # 1. 初始化 I2C (舵机用)
        self.ch347.CH347I2C_Set(self.dev_idx, 1)  # 100KHz
        self._init_pca9685()

        # 2. 初始化 GPIO (电磁铁用)
        self.ch347.CH347GPIO_Set.restype = c_bool
        self.ch347.CH347GPIO_Set.argtypes = [c_ulong, c_ubyte, c_ubyte, c_ubyte]
        self.gpio2_mask = 0x04  # 物理引脚 10
        self.set_magnet(False)  # 默认断开电磁铁

    def _write_reg(self, reg, val):
        buf = (c_ubyte * 3)((self.i2c_addr << 1), reg, val)
        if not self.ch347.CH347StreamI2C(self.dev_idx, 3, byref(buf), 0, None):
            raise IOError("CH347 register write failed")

    def _init_pca9685(self):
        self._write_reg(0x00, 0x11)  # Sleep
        time.sleep(0.05)
        self._write_reg(0xFE, 121)  # 50Hz
        self._write_reg(0x00, 0x21)  # Restart
        time.sleep(0.05)

    def set_pwm_us(self, channel, us):
        cfg = config.SERVO_CONFIG[channel]
        safe_us = max(cfg['min'], min(cfg['max'], us))
        tick = int((safe_us / 20000.0) * 4096)
        base_reg = 0x06 + 4 * channel
        data = (c_ubyte * 6)((self.i2c_addr << 1), base_reg, 0, 0, tick & 0xFF, (tick >> 8) & 0xFF)
        if not self.ch347.CH347StreamI2C(self.dev_idx, 6, byref(data), 0, None):
            raise IOError("CH347 PWM write failed")

    # === 新增：电磁铁控制接口 ===
    def set_magnet(self, on: bool):
        """控制 GPIO2 输出高低电平"""
        data = self.gpio2_mask if on else 0x00
        ok = self.ch347.CH347GPIO_Set(self.dev_idx, self.gpio2_mask, self.gpio2_mask, data)
        if not ok:
            raise IOError("CH347 electromagnet command failed")

    def close(self):
        """释放资源"""
        try:
            self.set_magnet(False)
        finally:
            self.ch347.CH347CloseDevice(self.dev_idx)
