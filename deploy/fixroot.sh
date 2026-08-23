#!/bin/bash
set -e
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
grep -q '^PermitRootLogin' /etc/ssh/sshd_config || sed -i '$a PermitRootLogin prohibit-password' /etc/ssh/sshd_config
if [ -d /etc/ssh/sshd_config.d ]; then
  grep -rl 'PermitRootLogin' /etc/ssh/sshd_config.d/ 2>/dev/null | xargs -r sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/'
fi
sshd -t
systemctl restart ssh
echo FIXROOT_DONE
grep -i PermitRootLogin /etc/ssh/sshd_config
