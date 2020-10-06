repository for the course project


# TO TRAIN TWOWAYNETS

Your directory sturcture should be:

ADARI/
	....
mmml\_f20 (this)/
	....
TwoWayNets (clone from https://github.com/ASchneidman/TwoWayNets)/
	....

cd into this directory (mmml\_f20)
Run script
./run\_training\_and\_embeddings.sh

Note: If you issue a keyboard interrupt (ctrl+C) while training, the script will capture it and output the current model weights with a unique name as a .pth file, and the embeddings for the test set. This will take a few minutes, just let it finish.
