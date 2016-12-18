if [ -z "$1" ]; then
  echo "Usage: sh enable-ssh-tunnel.sh [ip-address]"
  echo ""
  exit 1
fi
ssh -f -N -M -S /tmp/carnd.sock -L 4567:localhost:4567 carnd@$1
