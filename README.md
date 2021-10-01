# Seekar Target Classifier

Seekar Target Classifier is an API designed to locate and classify objects of interest in an image. It is optimized for 
noisy images such as those found in trail camera, drone footage, or satellite image datasets. The API is a python module
that contains a machine learning model that facilitates recognition and location of the object within an image.

Where applicable, open-sourced work is cited. All other work is directly built and copyrighted under Seekar Technologies, 
LLC and its affiliates.

## Running SeekarTargetClassifier
1. **Ensure Python 3.7 is installed on your computer.**

2. **cd into top-most parent directory of the project.**  `SeekarTargetClassifier` should be the top-most parent directory.

3. **Run the command `pip install -r requirements.txt`.**

4. **Run the command `python -m SeekarTargetClassifier`.** This will run the program as a module.

5. **Wait for the statement `SeekarTargetClassifier/Model/....loaded in X.XX seconds` to appear.** At this point, the 
model is fully loaded. This long boot-up sequence only happens once upon init.

6. **Type the command `wolf.jpeg` into the command line.** The command line waits for the user to type the image in. The
image must be added to the parent-most directory. A demo image called `wolf.jpeg` has been added for demonstrative purposes,
but any image may be added.

7. **A JSON response is returned to the command line.** The response contains the classification label, confidence, bounding box
coordinates of the located object, and image dimensions.

8. **Detection data is also written to the subdirectory `DETECTION_DATA\DETECTION_RESULTS`.** In there, there is a file called `DETECTION_RESULTS.txt` which contains
bounding box coordinates of the target bounds `bbox`, the name of the target `label`, the confidence level in the prediction
as a percentage `confidence`, the width of the image `im_width`, and the height of the image `im_height`.**  The image height
and width are important to note since the bounding box is dependent on the aspect ratio of the image. Additionally, there
is an image called `DETECTION_RESULTS.txt` which is the original image annotated with prediction bounding boxes around
recognized targets (if any). I incorporated the files in the `DETECTION_RESULTS` directory for your use in case it is
easier to grab data from a `.txt` file versus the console output. The console output is a direct duplicate of the text
file contents.

9. **The program will terminate after one iteration if the `SHOULD_REPEAT` parameter in `__main__.py` is set to `FALSE`.**
If you want it to run in a loop, change that setting to `TRUE`. Then, to run the classification again, simply type in 
the name of another image in the parent directory. The program waits for another image name to classify in the command 
line. Type in `x` or simply hit `enter` to terminate.


###Notes: 
1) There are some statements  printed to the console that we cannot disable. These are output from the `Tensorflow`
and `Keras` libraries. However, if you pull the last 5 lines of the console output, you should be able grab the data less
any debugging statements by the libraries. We are working to rid the program of Tensorflow.
-------------------------------------------------------------------------------
**Copyright 2018-2021 Seekar Technologies, LLC. All Rights Reserved.**

Seekar Technologies maintains the intellectual rights to this software
package, code files, and associated documentation files (the "Software").
The above copyright notice  notice shall be included in all copies or substantial
portions of the Software. Notwithstanding the foregoing, no person, entity, or 
organization may use, copy, modify, merge,
publish, distribute, sublicense, create a derivative work, and/or
sell copies of the Software in any work or form that is designed, intended,
or marketed for any purpose outside of Seekar Technologies, LLC. 
Permission for such use, copying, modifying,
merging, publishing, distributing, sub-licensing, creating
derivative works, or sale is expressly withheld. Seekar Technologies reserves
the right to pursue legal action should these permissions be violated.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.