#!/bin/bash

# play like a pro (or trying it...thanks master)
#http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
CLEAR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "$CLEAR_DIR"
echo "Entering directory \"$CLEAR_DIR\"."

# Define a function to count files of different extensions. Local variable "$1"
# must be and extension, so starting with a "."
count_files () {
	if [ "$1"=="."* ]
	then
		ls -1 *"$1" | wc -l
	else
		echo "Dude, you have to insert just an extension"
		exit 1
	fi

} 

# Check directory "./build" to be removed
BUILD="build"

if [ -d "$BUILD" ]
then
	echo "Removing directory \"$BUILD\"."
	rm -RI "$BUILD"
	if [ ! -d "$BUILD" ]
	then
		echo "Removed directory \"$BUILD\"."
	else
		echo "Directory \"$BUILD\" not removed."
	fi
else
	echo "No directory \"$BUILD\" found. Nothing to do."
fi

# Check if there are some ".extension" files...
EXTENSIONS=( ".cpp" ".h" ".pvtu" ".vtu" ".so")

for EXTENSION in "${EXTENSIONS[@]}"
do
	COUNT=$( count_files "$EXTENSION" )

	# ...and if there are, remove them...asking before, of course
	if [ "$COUNT" != 0 ]
	then
		echo "Found $COUNT files with \"$EXTENSION\" extension."
		rm -i *"$EXTENSION"
	else
		echo "No file with \"$EXTENSION\" extension."
	fi
		
done
