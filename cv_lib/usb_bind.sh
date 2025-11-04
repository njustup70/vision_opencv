#!/bin/bash
echo $1 | sudo tee /sys/bus/pci/drivers/xhci_hcd/bind
