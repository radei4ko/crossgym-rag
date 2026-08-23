#!/bin/bash
set -e
mkdir -p /root/.ssh
printf '%s\n' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE8P9vVAv8TbCzOUN+4+P3ZNNyUUCXCUalcKM2UQuYbp trainer-booking-vps' > /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
cat /root/.ssh/authorized_keys
echo SETUP_DONE
