rsync \
  --verbose \
  --recursive \
  --size-only \
  --exclude="*.DS_Store" \
  --include="IMG/center_*" \
  --include="driving_log.csv" \
  ./data \
  carnd@$1:/home/carnd/code/behavioral-cloning

# 54.67.76.6
