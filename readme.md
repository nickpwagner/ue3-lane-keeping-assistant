# Lane Keeping and Distance Control Simulation C++ with Unreal Engine (Tronis)

## Summary
The following only decribes the implemented logic. The environment and physics are modeled and simulated via Unreal Engine.

### Lane Detection

The lane detection process comprises five steps:

1. **hlsDetector**: A color mask is created using the HSL color space to detect white and yellow lanes, making it more stable.
2. **edgeDetector**: Corners and edges are detected through blurring, grayscale conversion, and the Canny algorithm.
4. A region of interest (ROI) is cropped from the mask by cutting out unnecessary areas, such as the hood and sky.
5. In the main lane detection step, the Hough transformation is used to recognize the lane lines. Left and right lane markings are identified based on their slopes, and straight lines are drawn using linear fitting.

### Vehicle Control

Vehicle control involves two parts:

In the lateral control, the vehicle's position within the lane is continuously compared with the target position, and steering is regulated using a PI controller.

In the longitudinal control, two cases are distinguished: CC (Cruise Control) and ACC (Adaptive Cruise Control). CC regulates speed using a PI controller, while ACC adjusts the speed to maintain a desired distance from the front vehicle.


## Detailed Description

The algorithm can be roughly divided into two parts: lane detection and vehicle control. Most of the required functions are called from the `DetectLanes()` function and are numbered with comments. Only the speed and distance control are called from the `getData()` function every time Tronis updates the speed.

The results are the longitudinal control signal (`acc_norm`), and the lateral control signal (`steer_norm`). Both are limited to values between -1 and 1.

### Lane Detection

The lane detection consists of five steps.

1. In the `hlsDetector` function, a color mask is created based on the HSL color space. RGB and HSV were also compared, but HSL is particularly suitable for white and yellow. White and yellow lanes (e.g., in construction zones) are typical, so yellow was directly implemented with white. Using HSL for white lines makes the lane detection more stable. A high lightness value is required for white lines, and a specific hue range and saturation for yellow lines. The resulting masks for white and yellow are binarized. To switch between yellow and white lanes, the number of non-zero mask entries is counted for both. The mask with more non-zero entries is used for further processing.

2. In the `edgeDetector` function, corners and edges are detected. This includes blurring using Gaussian convolution to reduce noise. The image is then converted to grayscale to facilitate edge detection using the Canny algorithm. The Canny algorithm is based on lower and upper threshold values, which can be adjusted during runtime using a trackbar (slider) along with other values. The detected edges are then enlarged using dilation based on a 3x3 kernel. Alternatively, edge detection using the Sobel operator is also programmed, and this can be switched using the `EDGEMODE` flag. However, Canny's performance is significantly better.

3. The color mask and edge mask are combined using a logical AND operation. This keeps only the entries that were detected both as colors and edges, filtering out unnecessary entries, such as grass adjacent to the road. This allows for a larger region of interest to be used, as not everything needs to be cropped, and more of the lane is retained in curves.

4. The region of interest (ROI) is then cropped from the mask. A polygon is drawn to cut out parts of the mask corresponding to the hood, the sky, and the areas outside the lane on the left and right sides ("fillPoly"). Another logical AND operation is applied with the previous result.

5. In the fifth and largest step of lane detection, the actual lane lines are recognized using the "houghLineDetector" function. The probabilistic Hough transformation is used, which allows the detection of simple geometric objects like lines and their scaling. Continuous structures are not required to a certain extent, which helps with detecting lane markings that might be dashed. The result is many detected lines, each given with a start and end point, which are then divided into left and right lane markings based on their slopes. First, too small slopes are filtered out using the absolute value, and then negative slopes are assigned to the left lane while positive slopes are assigned to the right lane and stored in separate lists. Additionally, the offset of the straight lines is calculated. If only one lane is detected, the last successfully detected lane markings are used. Otherwise, the median of the slopes/offsets in both lists is calculated and used to draw the recognized straight lines. Median is used as it is more robust against outliers than the average. The program can switch between median and average fitting using the "SLOPE_MEDIAN" flag. With the known slope and offset of the left and right lane lines, the appropriate X value can be calculated for the desired Y value. To make the detected lane lines more stable for robust control later, the last "LANE_MEM_MAX" detected lane lines are stored, and the average is then calculated. Alternatively, a polynomial fitting can be used in the program by switching the "POLY" flag, but the linear fitting is more stable and better optimized.

Optionally, a Bird's Eye View Transformation can be activated. However, in the program, it is only used for visualization and not for control, as it is too unstable for curves in the current implementation.

### Vehicle Control

The vehicle control consists of two parallel parts: longitudinal control (`accelerationControl`) and lateral control (`steeringControl`).

In the lateral control, the target position in X-coordinates within the lane is continuously compared with the current position, and the steering is regulated based on this comparison. Using a configurable Y-value, which represents the distance at which the X-value should lie for the target position, the midpoint of the lane in the X-direction is calculated. This requires dividing the X-values of both lane markings at this distance by two. Subtracting the current position in the X-direction from this midpoint gives the control difference. A PI controller is used for regulation, which determines the steering input. Additionally, the control value is normalized to a range between -1 and 1. This works well here because the possible min and max values of the control difference are known. With an image width of 580 pixels in the X-direction, the control difference can be at most half the image width, i.e., 290 pixels. The maximum steering input would then be 1, and normalization is done using 290 pixels. This normalization leads to very robust behavior, as no values need to be clipped at less than -1 or greater than 1, and the entire value range can be used for control.

In the longitudinal control, two cases are distinguished:

**Case 1: CC (Cruise Control)**

If there is no vehicle ahead or it is still far enough away, the `VelocityControl` function is called, which is responsible for speed regulation. It calculates the difference between the desired speed (`vel_t`) and the current actual speed (`vel_c`). This difference is then regulated using another PI controller. However, the control parameters for specific speeds vary. Instead of programming a large lookup table, these values have been tested and noted for the "GenericHatchback" in Tronis. Since the min and max values of the control difference can vary depending on the mode and speed limit, no normalization to a range of -1 to 1 is done here. Values below -1 or above 1 are simply clipped. If the calculated acceleration is negative and the vehicle speed is very low, the acceleration is set to zero to prevent driving backward. To prevent the integral part of the control from preventing coming to a complete stop, it is proportionally reduced when the speed is below 20 km/h and the vehicle is close to the front vehicle. If the current speed is also below 2 km/h, the vehicle is considered stationary.

**Case 2: ACC (Adaptive Cruise Control)**

If a vehicle is driving at a distance less than "DIST_ACT" meters ahead, the Adaptive Cruise Control mode is activated using the `accelerationControl()` function. This function calculates a desired speed that leads to the desired distance from the front vehicle. The desired distance changes dynamically and is set to half of the current vehicle's speed. If the speed is less than 20 km/h, it is fixed at 10 meters. Again, a PI controller is used, this time with the current speed as an offset. In braking scenarios, the offset is subtracted, and it has no effect.

A unique feature is that the time since the last call is measured and multiplied with the I-term increment. As Tronis sends updates irregularly, adjusting the I-term proportionally to the elapsed time leads to higher control stability. The I-term can influence the speed by a maximum of 2 km/h. Similar to the CC function, the I-term is proportionally reduced when the speed is below 20 km/h and when the vehicle is close to the front vehicle. Negative target speeds are set to zero. If Tronis does not provide a bounding box for an extended period or if the front vehicle disappears (e.g., during a lane change), this is detected using a watchdog, and ACC is deactivated, while CC is activated.

If the desired speed is higher than the set maximum speed "VEL_TAR," it will be limited to that value. If the actual distance falls below 5 meters despite regulation, emergency braking will be executed. The desired speed serves as input again for the same function used for the Cruise Control function, creating a cascaded controller. The distance control is robust against multiple front vehicles and even vehicles in adjacent lanes. In the `processBox()` function, appropriate boxes are determined based on size and position, and a single relevant box is calculated using the smallest distance. The deviation in the Y-direction is used to exclude vehicles from adjacent lanes. Additionally, the position of the BoundingBox sensor in Tronis must be compensated for, as it always assumes the center of the vehicles.

The acceleration request and steering request are communicated with Tronis through a TCP socket connection.
