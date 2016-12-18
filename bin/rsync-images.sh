rsync \
  --verbose \
  --recursive \
  --ignore-existing \
  --exclude="IMG/right*" \
  --exclude="IMG/left*" \
  --include="IMG/center_*" \
  ./data/IMG \
  carnd@$1:/home/carnd/code/behavioral-cloning/data/IMG

# 54.67.76.6
