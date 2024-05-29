# SYNAGEN: SYNthetic image Anomaly GENerator
SYNAGEN enables users to generate synthetic datasets by inserting synthetic anomalies with vast variances in size, shape, and texture into images. A detailed description of SYNAGEN can be found in the paper "Generation of Synthetic Image Anomalies for Analysis" in the ISPR2024 proceedings. 
## Installation
Clone this repository. SYNAGEN was tested with Python versions 3.8.19 and 3.11.4.
``` shell
# go to SYNAGEN folder
cd /.../synagen
# pip install required packages
pip install -r requirements.txt
```
## Configuration Files
The "config" folder contains  example configuration YAML files. These files define parameters for the synthetic anomaly generation and include documentation for each parameter. More detailed explanations of the anomaly generation process can be found in the paper "Generation of Synthetic Image Anomalies for Analysis" in the ISPR2024 proceedings. 
## Anomaly Generation
After successful installation, SYNAGEN is ready to generate anomalies. Here are a few example commands demonstrating how SYNAGEN can be used. The following line is the command and config file used in the original paper:
``` shell
# [input dataset] should be replaced with a path to the directory containing the original images.
# [output dataset] is the path to the directory where SYNAGEN saves the synthetic anomaly images.
python SYNAGEN.py config/config.yaml [input dataset] [output dataset]
```
### Predefined Textures
Instead of using the proposed pixel manipulation methods, SYNAGEN can also replace regions of the original image with predefined textures. The following line is the command and config file to accomplish this:
``` shell
# [input dataset] should be replaced with a path to the directory containing the original images (PNG or JPG).
# [output dataset] is the path to the directory where SYNAGEN saves the synthetic anomaly images (png or jpg).
# [texture dataset] is the path to the directory containing the texture images (PNG or JPG).
python SYNAGEN.py config/configTextures.yaml [input dataset] [output dataset] --textures [texture dataset]
```
### Positional Constraints
When SYNAGEN should place anomalies only in some areas of an image, the user needs to define these regions in a separate file for each image within the [input dataset] directory. This mask file needs to be of the same type (png or jpg) as the according image and have the same name with an "_mask" extension (e.g., the mask file for the image "exampleImage1.png" needs to be named "exampleImage1_mask.png"). Potential anomalous regions should be white, and regions where anomalies are not allowed must be black. The following line is the command to generate such anomalies with positional constraints:
``` shell
# [input dataset] should be replaced with a path to the directory containing the original images.
# [output dataset] is the path to the directory where SYNAGEN saves the synthetic anomaly images.
python SYNAGEN.py config/config.yaml [input dataset] [output dataset] --masked
```
These positional masks in SYNAGEN's implementation have priority over the size distribution defined in the config file. Suppose one of SYNAGEN's anomalies does not fit within a masked region. In that case, SYNAGEN alters only the pixels within the allowed regions, leading to a mismatch between the target size and the resulting size of the anomaly.
