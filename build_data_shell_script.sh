#!/bin/sh
#name_tring = find *.jpg
#the srting which in the equation must combined together

# find the path of images of 
#cur_path=$(cd `dirname $0`; pwd)
#jpg_file_path=$(find $cur_path -name *.jpg)
#png_file_path=$(find $cur_path -name *.png)

# counting the numbers of images of *.jpg format:
#count_img=0
#for str_ in $jpg_file_path; do
#   count_img=`expr $count_img + 1`
#done
#echo $count_img

cd AMOS_Data

# listing the class of images
list_class=$(ls)

# make dir for train and validation
mkdir train validation

# move the images to train dir and validation dir
for class_camera in $list_class; do
   # create the class_camera dir in the train and validation
   class_name=${class_camera##*/}
   cd train
   mkdir $class_name
   cd ..
   cd validation
   mkdir $class_name
   cd ..
 
   # move the image before 22/7/2013 to train and the other move to validationpointment
   cd $class_camera
   cd 2013.07
   cur_path=$(cd `dirname $0`; pwd) #currrent path
   jpg_file_path=$(find $cur_path -name *.jpg) #find image
   for str_ in $jpg_file_path; do
      file_name=$(basename $str_ .jpg)
      if ${file_name%%_*}<20130723; then
         mv str_ ~/czy/dataset/AMOS/AMOS_Data/train
      else
         mv str_ ~/czy/dataset/AMOS/AMOS_Data/validation
      fi
   done
done


