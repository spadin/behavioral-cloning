rsync \
  --verbose \
  --recursive \
  --size-only \
  --exclude="IMG/right*" \
  --exclude="IMG/left*" \
  --exclude="*.DS_Store" \
  --include="IMG/center_*" \
  --include="driving_log.csv" \
  ./data \
  carnd@54.67.76.6:/home/carnd/code/behavioral-cloning

# 54.67.76.6
