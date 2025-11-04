#!/bin/bash
# 接收USB Hub的PCI地址（如 0000:00:14.0）
echo $1 | sudo tee /sys/bus/pci/drivers/xhci_hcd/unbind
