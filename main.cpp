#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>

using namespace std;
using namespace cv;

class LaneAssistant
{
public:
    LaneAssistant()
    {
        /// call trackbar function in the constructor to directly open up
        trackbarDetectLanes();
    }

    bool processData( tronis::CircularMultiQueuedSocket& socket )
    {
        /// transmit the accerelation and steering commands to tronis
        string command = to_string( acc_norm ) + ";" + to_string( steer_norm );
        socket.send( tronis::SocketData( command ) );
        return true;
    }

protected:
    //_________________ Hyperparameters _________________/////
    bool POLY = false;         /// false = linearRegression; true = polynomialRegression
    bool SLOPE_MEDIAN = true;  /// false = average; true = median
    bool LANE_MEMORY =
        true;  /// false = only use most recent result; true = use average of past x results
    int LANE_MEM_MAX = 6;       /// number of past values, that get used for smoothing
    bool DILATION = true;       /// true = edge+color detector results get dilated
    string EDGEMODE = "canny";  /// "canny", "sobel"
    int Y = 300;                /// controller target line (smaller = earlier curve reaction
    int Y2 = 300;               /// detection top limit
    int Y1 = 580;               /// detection bottom limit
    float STEER_P = 1;          /// P-Factor for STEER PID controller
    float STEER_I = 0.00025;    /// I-Factor for STEER PID controller
    float STEER_D = 0;          /// D-Factor for STEER PID controller
    float STEER_MAX =
        290;  /// max pixel diff between x_curr and x_target (580/2 for max) to normalize
    double VEL_TAR = 50;          /// Max velocity without preceeding vehicle
    int ACC_P = 45;               /// P-Factor for ACCELERATION PID controller
    int ACC_I = 125;              /// I-Factor for ACCELERATION PID controller
    int ACC_D = 0;                /// D-Factor for ACCELERATION PID controller
    float DIST_ACT = 100;         /// Distance at which an object becomes relevant for ACC in meters
    double DIST_TAR_SLOW = 10;    /// target distance to preceeding vehicle
    int DIST_P = 15;              /// P-Factor for DIST PID controller
    int DIST_I = 15;              /// I-Factor for DIST PID controller
    int DIST_D = 0;               /// D-Factor for DIST PID controller
    int DIST_WATCHDOG_MAX = 150;  /// checks if tronis still sends distance updates
    bool moveWindows = true;      /// moves imshow windows to the left of the screen

    //_________________ Variables _________________/////
    // tronis socket variables
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    tronis::BoxDataSub boxes_;
    double ego_velocity_;
    // Detection
    string image_name_;
    Mat image_orig;                                       /// original tronis image
    Mat img_gray, img_edge, img_hsv, img_hls;             /// edge detection
    Mat mask_white_hsv, mask_yellow_hsv;                  /// HSV: color white/yellow mask
    Mat mask_white_hls, mask_yellow_hls;                  /// HSV: color white/yellow mask
    Mat mask_final;                                       /// currently active color mask
    Mat img_edge_color, img_edge_color_cone;              /// edge+color(+cone crop) detection
    Mat img_bev, M_bev, Minv_bev;                         /// bird's eye view
    Mat img_hough_all, img_hough_linear, img_hough_poly;  /// line detection
    Mat img_steer;                                        /// final steer cmd overlayed on image
    /// Steering
    Point leftLane[2];                /// [0] = BL, [1] = TL
    Point rightLane[2];               /// [0] = BR, [1] = TR
    Point leftLane_poly[2];           /// same for poly fit
    Point rightLane_poly[2];          /// same for poly fit
    double steer_norm, acc_norm = 0;  /// final cmd sent to tronis
    double avg_slope_l, avg_slope_r, avg_intercept_l, avg_intercept_r = 0;  /// avg line values
    vector<double> mem_slope_l, mem_slope_r, mem_intercept_l,
        mem_intercept_r;           /// line memory to average over time
    double steer_error_P_old = 0;  /// steering P at t-1
    double steer_error_I_sum = 0;  /// steering I sum
    int mem_counter = 0;           /// avoids reading beyond mem list limits
    // Acceleration
    double vel_error_P_old = 0;  /// velocity P at t-1
    double vel_error_I_sum = 0;  /// velocity I sum
    // Distance Control
    double dist_curr = 101;       /// current distance to preceeding vehicle
    double dist_curr_old = 0;     /// distance at t-1
    double dist_error_P_old = 0;  /// to check if tronis still updates boxes
    double dist_error_I_sum = 0;  /// distance I sum
    int dist_watchdog = 0;        /// to check if tronis still updates boxes
    double vel_tar = 0;           /// calculated target velocity
    double t_dist_diff = 0;       /// time since last function call
    chrono::time_point<chrono::steady_clock> t_dist_start;  /// stop watch
    chrono::time_point<chrono::steady_clock> t_dist_stop;
    // trackbar adjustable variables
    /// Canny / Sobel
    int canny_low = 200;
    int canny_high = 250;
    int sobel_thresh = 210;
    /// HSV
    int hmin = 60, smin = 15, vmin = 50;
    int hmax = 180, smax = 199, vmax = 200;
    /// HLS White
    int hls_h_min = 0, hls_l_min = 100, hls_s_min = 0;
    int hls_h_max = 180, hls_l_max = 155, hls_s_max = 255;
    /// HLS Yellow
    int hls_h_min_y = 10, hls_l_min_y = 0, hls_s_min_y = 75;
    int hls_h_max_y = 23, hls_l_max_y = 255, hls_s_max_y = 255;
    /// Hough
    double pi = 3.14159265358979323846;
    int rho = 1;
    int tb_theta = pi / 60;
    int thresh = 100;
    int minLine = 25;
    int maxGap = 25;
    int slope_thr = 20;
    /// Polyfit
    int order = 2;

    void detectLanes()
    {  /// ---------- 1. COLOR THRESHOLDING ---------- ///
        // cout << "1. COLOR THRESHOLDING" << endl;
        hlsDetector( image_orig );
        // hsvDetector( image_orig );

        /// ---------- 2. EDGE DETECTION ---------- ///
        // cout << "2. EDGE DETECTION" << endl;
        edgeDetector( image_orig, EDGEMODE );
        /// Edge + Color Fusion
        crop( mask_yellow_hls );
        crop( mask_white_hls );
        mask_final = mask_white_hls;
        /// if more yellow lanes than white, activate yellow
        if( countNonZero( mask_white_hls ) < countNonZero( mask_yellow_hls ) )
        {
            // cout << "YELLOW ACTIVE" << endl;
            mask_final = mask_yellow_hls;
        }
        bitwise_and( img_edge, mask_final, img_edge_color );

        /// ---------- 3. ROI CROP ---------- ///
        // cout << "3. ROI CROP" << endl;
        crop( img_edge_color );

        /// ---------- 4. LINE DETECTION ---------- ///
        // cout << "4. LINE DETECTION" << endl;
        houghLineDetector( img_edge_color_cone );

        /// ---------- 5. BIRD'S EYE VIEW ---------- ///
        // cout << "5. BIRD'S EYE VIEW" << endl;
        birdsEyeView( img_hough_linear );

        /// ---------- 5. STEERING CONTROL ---------- ///
        // cout << "5. STEERING CONTROL" << endl;
        if( POLY )
        {
            steeringControl( img_hough_poly );
        }
        else
        {
            steeringControl( img_hough_linear );
        }

        /// ---------- 6. ACCELERATION CONTROL ---------- ///
        // moved - now gets called whenever a velocity update is received from tronis (in getData())

        /// ---------- 7. PRINT RESULTS ---------- ///
        // cout << "7. PRINT RESULTS" << endl;
        print();
        waitKey( 1 );
    }

    void print()
    {
        namedWindow( "Edge Detection", ( WINDOW_NORMAL ) );
        imshow( "Edge Detection", img_edge );

  //      namedWindow( "HLS Detection White", ( WINDOW_NORMAL ) );
  //      imshow( "HLS Detection White", mask_white_hls );
  //      
		//namedWindow( "HLS Detection Yellow", ( WINDOW_NORMAL ) );
  //      imshow( "HLS Detection Yellow", mask_yellow_hls );

        namedWindow( "Color Mask", ( WINDOW_NORMAL ) );
        imshow( "Color Mask", mask_final );

        namedWindow( "Edge+Color Fusion", ( WINDOW_NORMAL ) );
        imshow( "Edge+Color Fusion", img_edge_color_cone );

        namedWindow( "Line Detection", ( WINDOW_NORMAL ) );
        imshow( "Line Detection", img_hough_all );

        // namedWindow( "Poly Line Detection", ( 600, 400 ) );
        // imshow( "Poly Line Detection", img_hough_poly );

        namedWindow( "Steering Controller", ( WINDOW_NORMAL ) );
        imshow( "Steering Controller", img_steer );

        // namedWindow( "BEV", ( WINDOW_NORMAL ) );
        // imshow( "BEV", img_bev );

        /// moves the windows to the right screen side once
        if( moveWindows )
        {
            moveWindow( "Edge Detection", 0, 0 );
            resizeWindow( "Edge Detection", 275, 275 );
            moveWindow( "Color Mask", 0, 275 );
            resizeWindow( "Color Mask", 275, 275 );
            moveWindow( "Edge+Color Fusion", 0, 550 );
            resizeWindow( "Edge+Color Fusion", 275, 275 );
            moveWindow( "Steering Controller", 275, 0 );
            resizeWindow( "Steering Controller", 500, 600 );
            moveWindow( "Line Detection", 275, 600 );
            resizeWindow( "Line Detection", 500, 200 );

            moveWindows = false;
        }
    }
    ///________________LANE DETECTION_________________/////
    void trackbarDetectLanes()
    {
        // namedWindow( "Edge Detection" );
        // createTrackbar( "Canny low", "Edge Detection", &canny_low, 255 );
        // createTrackbar( "Canny high", "Edge Detection", &canny_high, 255 );
        // createTrackbar( "Soeb thresh", "Edge Detection", &sobel_thresh, 255 );

        // namedWindow( "HSV Detection" );
        // createTrackbar( "hue min", "HSV Detection", &hmin, 180 );
        // createTrackbar( "hue max", "HSV Detection", &hmax, 180 );
        // createTrackbar( "sat min", "HSV Detection", &smin, 255 );
        // createTrackbar( "sat max", "HSV Detection", &smax, 255 );
        // createTrackbar( "val min", "HSV Detection", &vmin, 255 );
        // createTrackbar( "val max", "HSV Detection", &vmax, 255 );

        //namedWindow( "HLS Detection White", WINDOW_AUTOSIZE );
        //createTrackbar( "hue min", "HLS Detection White", &hls_h_min, 180 );
        //createTrackbar( "hue max", "HLS Detection White", &hls_h_max, 180 );
        //createTrackbar( "light min", "HLS Detection White", &hls_l_min, 255 );
        //createTrackbar( "light max", "HLS Detection White", &hls_l_max, 255 );
        //createTrackbar( "sat min", "HLS Detection White", &hls_s_min, 255 );
        //createTrackbar( "sat max", "HLS Detection White", &hls_s_max, 255 );
        
		//namedWindow( "HLS Detection Yellow", WINDOW_AUTOSIZE );
  //      createTrackbar( "hue min", "HLS Detection Yellow", &hls_h_min_y, 180 );
  //      createTrackbar( "hue max", "HLS Detection Yellow", &hls_h_max_y, 180 );
  //      createTrackbar( "light min", "HLS Detection Yellow", &hls_l_min_y, 255 );
  //      createTrackbar( "light max", "HLS Detection Yellow", &hls_l_max_y, 255 );
  //      createTrackbar( "sat min", "HLS Detection Yellow", &hls_s_min_y, 255 );
  //      createTrackbar( "sat max", "HLS Detection Yellow", &hls_s_max_y, 255 );

        namedWindow( "line detection", WINDOW_AUTOSIZE );
        createTrackbar( "rho", "line detection", &rho, 10 );
        createTrackbar( "theta", "line detection", &tb_theta, 100 );
        createTrackbar( "thresh", "line detection", &thresh, 300 );
        createTrackbar( "MinLength", "line detection", &minLine, 100 );
        createTrackbar( "MaxGap", "line detection", &maxGap, 50 );
        createTrackbar( "slope thr l/r", "line detection", &slope_thr, 100 );

        // namedWindow( "Acceleration", ( 1000, 200 ) );
        // createTrackbar( "P", "Acceleration", &ACC_P, 300 );
        // createTrackbar( "I", "Acceleration", &ACC_I, 300 );
        // createTrackbar( "D", "Acceleration", &ACC_D, 300 );

        // namedWindow( "Distance", ( 1000, 200 ) );
        // createTrackbar( "P", "Distance", &DIST_P, 100 );
        // createTrackbar( "I", "Distance", &DIST_I, 100 );
        // createTrackbar( "D", "Distance", &DIST_D, 100 );
    }
    void edgeDetector( Mat img, string mode )
    {
        /// Remove noise by blurring with a Gaussian filter
        GaussianBlur( img, img_edge, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );
        /// Convert the image to grayscale
        cvtColor( img_edge, img_gray, COLOR_BGR2GRAY );

        if( EDGEMODE == "canny" )
        {
            /// Apply Canny Edge Detectpr
            Canny( img_gray, img_edge, canny_low, canny_high );
            /// Dilate (increase size) of edges
            if( DILATION )
            {
                Mat kernel = getStructuringElement( MORPH_RECT, Size( 3, 3 ) );
                dilate( img_edge, img_edge, kernel );
            }
        }
        else if( EDGEMODE == "sobel" )
        {
            Mat grad_x, grad_y, grad;
            Mat abs_grad_x, abs_grad_y;
            int ksize = 3;
            int ddepth = CV_16S;
            int maxval = 255;
            int thresh_type = 3;

            /// Gradient X,Y
            Sobel( img_gray, grad_x, ddepth, 1, 0, ksize, BORDER_DEFAULT );
            Sobel( img_gray, grad_y, ddepth, 0, 1, ksize, BORDER_DEFAULT );
            /// converting back to CV_8U
            convertScaleAbs( grad_x, abs_grad_x );
            convertScaleAbs( grad_y, abs_grad_y );
            /// Total Gradient (approximate)
            addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0,
                         grad );  //(1-alpha)*f0 + beta*f1 + gamma
            // Threshold Image
            threshold( grad, grad, sobel_thresh, maxval, thresh_type );
            /// Dilate (increase size) of edges
            if( DILATION )
            {
                Mat kernel = getStructuringElement( MORPH_RECT, Size( 3, 3 ) );
                dilate( grad, img_edge, kernel );
            }
            imshow( "sobel", img_edge );
        }
        else
        {
            cout << "INVALID EDGEMODE" << endl;
        }
    }
    void hsvDetector( Mat img )
    {
        /// HSV - thresholding
        /// Hue = color (0-180)
        /// Saturation = color strength (0-255)
        /// Value = dark to bright (0-255)

        /// White lines
        cvtColor( img, img_hsv, COLOR_BGR2HSV );
        Scalar white_lower( hmin, smin, vmin );
        Scalar white_upper( hmax, smax, vmax );
        inRange( img_hsv, white_lower, white_upper, mask_white_hsv );
        mask_white_hsv.convertTo( mask_white_hsv, CV_8UC1 );
        Mat kernel = getStructuringElement( MORPH_RECT, Size( 3, 3 ) );
        dilate( mask_white_hsv, mask_white_hsv, kernel );
        image_orig.copyTo( img_hsv, mask_white_hsv );
    }
    void hlsDetector( Mat img )
    {
        /// HLS - thresholding
        /// Hue = color (0-180)
        /// Lightness = dark to bright (0-255)
        /// Saturation = color strength (0-255)

        /// White lines
        cvtColor( img, img_hls, COLOR_BGR2HLS );
        Scalar white_lower( hls_h_min, hls_l_min, hls_s_min );
        Scalar white_upper( hls_h_max, hls_l_max, hls_s_max );
        inRange( img_hls, white_lower, white_upper, mask_white_hls );
        mask_white_hls.convertTo( mask_white_hls, CV_8UC1 );
        Mat kernel = getStructuringElement( MORPH_RECT, Size( 3, 3 ) );
        dilate( mask_white_hls, mask_white_hls, kernel );
        // image_orig.copyTo( img_hls, mask_white_hls );
        // imshow( "WHITE LINES", img_hls );

        /// Yellow lines
        cvtColor( img, img_hls, COLOR_BGR2HLS );
        Scalar yellow_lower( hls_h_min_y, hls_l_min_y, hls_s_min_y );
        Scalar yellow_upper( hls_h_max_y, hls_l_max_y, hls_s_max_y );
        // Scalar yellow_lower( hls_h_min, 0, 75 );
        // Scalar yellow_upper( 23, 255, 255 );
        inRange( img_hls, yellow_lower, yellow_upper, mask_yellow_hls );
        mask_yellow_hls.convertTo( mask_yellow_hls, CV_8UC1 );
        dilate( mask_yellow_hls, mask_yellow_hls, kernel );
        // image_orig.copyTo( img_hls, mask_yellow_hls );
        // imshow( "YELLOW LINES", img_hls );
    }
    Mat crop( Mat& img )
    {
        int img_height = img.size().height;
        int img_width = img.size().width;
        /// mask image with polygon
        Mat mask_crop( img_height, img_width, CV_8UC1, Scalar( 0 ) );
        /// create cone shape
        vector<Point> pts;
        vector<vector<Point>> v_pts;
        pts.push_back( Point( 0, img_height / 2 + 90 ) );
        pts.push_back( Point( img_width / 2, img_height / 2 ) );
        pts.push_back( Point( img_width, img_height / 2 + 90 ) );
        pts.push_back( Point( img_width, img_height - 60 ) );
        pts.push_back( Point( img_width * 0.7, img_height - 125 ) );
        pts.push_back( Point( img_width * 0.3, img_height - 125 ) );
        pts.push_back( Point( 0, img_height - 60 ) );
        v_pts.push_back( pts );
        /// add cone to mask
        fillPoly( mask_crop, v_pts, 255 );

        // apply mask to image
        bitwise_and( img, mask_crop, img_edge_color_cone );
        bitwise_and( img, mask_crop, img );
        // image_orig.copyTo( temp, mask_crop2 ); //if color channel and mask
        return img_edge_color_cone;
    }
    void houghLineDetector( Mat img )
    {
        /// adapt trackbar int to double value
        double theta = ( (double)tb_theta ) / 1000;
        Mat hough_all( img.size(), CV_8UC4, Scalar( 0, 0, 0, 1 ) );
        Mat hough_curve( img.size(), CV_8UC4, Scalar( 0, 0, 0, 1 ) );
        /// will hold the results of the detection
        vector<Vec4i> lines;

        // STEP 1: Detect and draw all lines into the image
        /// runs the hough transformation and detects lines
        HoughLinesP( img, lines, rho, CV_PI / 180, thresh, minLine, maxGap );
        /// Draw the lines - only required for visual check via imshow
        //     for( size_t i = 0; i < lines.size(); i++ )
        //     {
        //         Vec4i l = lines[i];
        //         line( hough_all, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 255, 0, 0, 1
        //         ), 3,
        //               LINE_AA );
        //     }
        //     cvtColor( hough_all, hough_all, COLOR_BGRA2RGBA );
        /// overlay lines on original image
        //     addWeighted( image_orig, 0.8, hough_all, 1.0, 0.0, img_hough_all );

        /// fit a line for each lane based on a polynomial with order 1
        LinearFit( lines );
        /// fit a line for each lane based on a polynomial with order 2
        PolynomialFit( lines );
    }
    void LinearFit( vector<Vec4i> lines )
    {
        vector<double> slope_l, slope_r;
        vector<double> intercept_l, intercept_r;
        Mat hough_all( image_orig.size(), CV_8UC4, Scalar( 0, 0, 0, 1 ) );
        // STEP 2: Lane Marking left/right
        /// go through lines and calculate if left or right via slope sign
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            Point p_start = Point( l[0], l[1] );
            Point p_end = Point( l[2], l[3] );
            /// slope m = (y2-y1)/(x2-x1)
            double slope = ( p_end.y - p_start.y ) / (double)( p_end.x - p_start.x );
            /// intercept b = y-m*x
            double intercept = p_start.y - slope * p_start.x;
            /// filter for fittin slopes (divided by ten to have int for trackbar)
            if( abs( slope ) > ( (double)slope_thr / 100 ) )
            {
                // cout << "SLOPE" << slope << endl;
                /// draw line for to check which ones remain via trackbars
                line( hough_all, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 255, 0, 0, 1 ),
                      3, LINE_AA );
                /// left lane marking
                if( slope < 0 )
                {
                    slope_l.push_back( slope );
                    intercept_l.push_back( intercept );
                }
                /// right lane marking
                else if( slope > 0 )
                {
                    slope_r.push_back( slope );
                    intercept_r.push_back( intercept );
                }
            }
        }
        /// overlay lines on original image
        cvtColor( hough_all, hough_all, COLOR_BGRA2RGBA );
        addWeighted( image_orig, 0.8, hough_all, 1.0, 0.0, img_hough_all );

        // STEP 3: Linear Fit
        /// two lanes found > update, else use old lanes by not updating
        if( !slope_l.empty() && !slope_r.empty() )
        {
            /// use median slope and intercept to be robust to outliers
            if( SLOPE_MEDIAN )
            {
                avg_slope_l = median( slope_l );
                avg_slope_r = median( slope_r );
                avg_intercept_l = median( intercept_l );
                avg_intercept_r = median( intercept_r );
            }
            /// use average slope and intercept lane
            else
            {
                avg_slope_l = accumulate( slope_l.begin(), slope_l.end(), 0.0 ) / slope_l.size();
                avg_slope_r = accumulate( slope_r.begin(), slope_r.end(), 0.0 ) / slope_r.size();
                avg_intercept_l =
                    accumulate( intercept_l.begin(), intercept_l.end(), 0.0 ) / intercept_l.size();
                avg_intercept_r =
                    accumulate( intercept_r.begin(), intercept_r.end(), 0.0 ) / intercept_r.size();
            }

            /// lane smoothing: update memory of past lane values and use average
            if( LANE_MEMORY )
            {
                /// fill memory with up to mem_max entries
                if( mem_slope_l.size() < LANE_MEM_MAX )
                {
                    mem_slope_l.push_back( avg_slope_l );
                    mem_slope_r.push_back( avg_slope_r );
                    mem_intercept_l.push_back( avg_intercept_l );
                    mem_intercept_r.push_back( avg_intercept_r );
                }
                /// if list is full, start updating from the first postition upwards
                else if( mem_slope_l.size() == LANE_MEM_MAX )
                {
                    mem_slope_l[mem_counter] = avg_slope_l;
                    mem_slope_r[mem_counter] = avg_slope_r;
                    mem_intercept_l[mem_counter] = avg_intercept_l;
                    mem_intercept_r[mem_counter] = avg_intercept_r;

                    if( mem_counter == LANE_MEM_MAX - 1 )
                    {
                        mem_counter = 0;
                    }
                    else
                    {
                        mem_counter++;
                    }
                }
                /// calculate average of memory
                avg_slope_l =
                    accumulate( mem_slope_l.begin(), mem_slope_l.end(), 0.0 ) / mem_slope_l.size();
                avg_slope_r =
                    accumulate( mem_slope_r.begin(), mem_slope_r.end(), 0.0 ) / mem_slope_r.size();
                avg_intercept_l =
                    accumulate( mem_intercept_l.begin(), mem_intercept_l.end(), 0.0 ) /
                    mem_intercept_l.size();
                avg_intercept_r =
                    accumulate( mem_intercept_r.begin(), mem_intercept_r.end(), 0.0 ) /
                    mem_intercept_r.size();
            }
        }

        // STEP 4: Calculate x-values based on vertical y-range
        /// x = (y-b)/m
        int x1_l = (int)( Y1 - avg_intercept_l ) / avg_slope_l;
        int x1_r = (int)( Y1 - avg_intercept_r ) / avg_slope_r;
        int x2_l = (int)( Y2 - avg_intercept_l ) / avg_slope_l;
        int x2_r = (int)( Y2 - avg_intercept_r ) / avg_slope_r;

        leftLane[0] = Point( x1_l, Y1 );   // [0] = BL
        leftLane[1] = Point( x2_l, Y2 );   // [1] = TL
        rightLane[0] = Point( x1_r, Y1 );  // [0] = BR
        rightLane[1] = Point( x2_r, Y2 );  // [1] = TR

        // STEP 5: overlay cropped original with ROI polygon and lines
        /// overlay lane ROI polygon
        img_hough_linear = Mat( image_orig.size(), CV_8UC4, Scalar( 0, 0, 0, 1 ) );
        Mat mask_laneFill = img_hough_linear.clone();
        vector<Point> pts;
        vector<vector<Point>> v_pts;
        pts.push_back( leftLane[1] );
        pts.push_back( rightLane[1] );
        pts.push_back( rightLane[0] );
        pts.push_back( leftLane[0] );
        v_pts.push_back( pts );
        fillPoly( mask_laneFill, v_pts, Scalar( 255, 0, 0, 1 ) );
        addWeighted( img_hough_linear, 1, mask_laneFill, 0.3, 0.0, img_hough_linear );
        /// overlay lane lines
        line( img_hough_linear, leftLane[0], leftLane[1], Scalar( 255, 255, 0, 1 ), 3, LINE_AA );
        line( img_hough_linear, rightLane[0], rightLane[1], Scalar( 255, 255, 0, 1 ), 3, LINE_AA );
        addWeighted( image_orig, 0.8, img_hough_linear, 1.0, 0.0, img_hough_linear );
    }
    void PolynomialFit( vector<Vec4i> lines )
    {
        /// Multiple Linear Regression (closed-form) to fit curves (only linear in parameters)
        // STEP 2: Lane Marking left/right
        /// go through lines and calculate if left or right via slope sign
        vector<Point> line_l, line_r;
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            Point p_start = Point( l[0], l[1] );
            Point p_end = Point( l[2], l[3] );
            /// slope m = (y2-y1)/(x2-x1)
            double slope = ( p_end.y - p_start.y ) / (double)( p_end.x - p_start.x );
            if( abs( slope ) > ( slope_thr / 100 ) )
            {
                /// left lane marking
                if( slope < 0 )
                {
                    line_l.push_back( Point( p_start.y, p_start.x ) );
                    line_l.push_back( Point( p_end.y, p_end.x ) );
                }
                /// right lane marking
                else if( slope > 0 )
                {
                    line_r.push_back( Point( p_start.y, p_start.x ) );
                    line_r.push_back( Point( p_end.y, p_end.x ) );
                }
            }
        }
        // STEP 3: Polynomial Fit
        Mat poly_coeff_l = PolynomialFit_( line_l );
        Mat poly_coeff_r = PolynomialFit_( line_r );
        Mat hough_curve( image_orig.size(), CV_8UC4, Scalar( 0, 0, 0, 1 ) );
        /// draw dots for lane markings (green left, blue right) for all 58 vertical pixels
        for( int y = 0; y < image_orig.size[1]; y++ )
        {
            /// apply calculated coefficients f = a0 + a1*x + a2*x^2
            double x_left = poly_coeff_l.at<double>( 0 ) + y * poly_coeff_l.at<double>( 1 ) +
                            y * y * poly_coeff_l.at<double>( 2 );
            double x_right = poly_coeff_r.at<double>( 0 ) + y * poly_coeff_r.at<double>( 1 ) +
                             y * y * poly_coeff_r.at<double>( 2 );
            /// draw dots as curve marks
            circle( hough_curve, Point( x_left, y ), 3, Scalar( 0, 255, 0 ), FILLED );
            circle( hough_curve, Point( x_right, y ), 3, Scalar( 255, 255, 0 ), FILLED );
        }
        // STEP 4: Calculate x-values based on vertical y-range
        double x_left_max = poly_coeff_l.at<double>( 0 ) + Y2 * poly_coeff_l.at<double>( 1 ) +
                            Y2 * Y2 * poly_coeff_l.at<double>( 2 );
        double x_left_min = poly_coeff_l.at<double>( 0 ) + Y1 * poly_coeff_l.at<double>( 1 ) +
                            Y1 * Y1 * poly_coeff_l.at<double>( 2 );
        double x_right_max = poly_coeff_r.at<double>( 0 ) + Y2 * poly_coeff_r.at<double>( 1 ) +
                             Y2 * Y2 * poly_coeff_r.at<double>( 2 );
        double x_right_min = poly_coeff_r.at<double>( 0 ) + Y1 * poly_coeff_r.at<double>( 1 ) +
                             Y1 * Y1 * poly_coeff_r.at<double>( 2 );

        leftLane_poly[0] = Point( x_left_min, Y1 );
        leftLane_poly[1] = Point( x_left_max, Y2 );
        rightLane_poly[0] = Point( x_right_min, Y1 );
        rightLane_poly[1] = Point( x_right_max, Y2 );

        // STEP 5: overlay cropped original with lines
        img_hough_poly = Mat( image_orig.size(), CV_8UC4, Scalar( 0, 0, 0, 1 ) );
        line( img_hough_poly, leftLane_poly[0], leftLane_poly[1], Scalar( 0, 255, 0, 1 ), 3,
              LINE_AA );
        line( img_hough_poly, rightLane_poly[0], rightLane_poly[1], Scalar( 255, 255, 0, 1 ), 3,
              LINE_AA );
        addWeighted( image_orig, 0.8, img_hough_poly, 1.0, 0.0, img_hough_poly );
    }
    Mat PolynomialFit_( vector<Point> points )
    {
        /// closed form solution with possibility to change order
        /// beta = parameters
        /// X = features x samples
        /// Y = label
        Mat X( points.size(), ( order + 1 ), CV_64F );
        Mat Y( points.size(), 1, CV_64F );

        for( int i = 0; i < X.rows; i++ )
        {
            for( int j = 0; j < X.cols; j++ )
            {
                X.at<double>( i, j ) = pow( points[i].x, j );
            }
        }
        for( int i = 0; i < Y.rows; i++ )
        {
            Y.at<double>( i, 0 ) = points[i].y;
        }
        Mat beta( ( order + 1 ), 1, CV_64F );
        if( X.data != NULL )
        {
            beta = ( X.t() * X ).inv() * X.t() * Y;
        }
        return beta;
    }
    double median( vector<double> vector )
    {
        /// find median value of vector
        size_t size = vector.size();
        sort( vector.begin(), vector.end() );
        if( size % 2 == 0 )
        {
            return ( vector[size / 2 - 1] + vector[size / 2] ) / 2;
        }
        else
        {
            return vector[size / 2];
        }
        return 0;
    }
    void birdsEyeView( Mat img )
    {
        /* NOT USED IN ALGORITHM, BUT CAN BE ACTIVATED FOR DISPLAYING PURPOSES */
        /// TL, TR, BR, BL (720x580)
        Point2f input_points[4] = {
            {356 - 80, float( Y2 )}, {356 + 80, float( Y2 )}, {720, float( Y1 )}, {0, float( Y1 )}};
        Point2f target_points[4] = {{356 - 80, float( Y2 )},
                                    {356 + 80, float( Y2 )},
                                    {356 + 80, float( Y1 )},
                                    {356 - 80, float( Y1 )}};
        M_bev = getPerspectiveTransform( input_points, target_points );
        Minv_bev = getPerspectiveTransform( target_points, input_points );
        warpPerspective( img, img_bev, M_bev, img.size() );
        /// display input/output points in images before and after warping
        for( int i = 0; i < 4; i++ )
        {
            // circle( img_hough_linear, input_points[i], 5, Scalar( 255, 255, 255 ), FILLED );
            // circle( img_bev, target_points[i], 5, Scalar( 255, 255, 255 ), FILLED );
        }
        /// cut off upper half (whole: 720x512)
        img_bev = img_bev( cv::Rect( 0, 270, img_bev.size().width, 58 ) );
    }
    void inv_birdsEyeView( Mat img )
    {
        /// inverts the BEV transformation
        Mat img_bev_inv;
        warpPerspective( img, img_bev_inv, Minv_bev, img.size() );
        // imshow( "Inverse BEV", img_bev_inv );
    }
    //_________________ STEERING _________________/////
    void steeringControl( Mat img )
    {
        img_steer = img;
        // 1. Find x_left and x_right with x=x1+(y-y1)/m
        double x_left = double( leftLane[0].x ) + double( Y - leftLane[0].y ) / avg_slope_l;
        double x_right = double( rightLane[0].x ) + double( Y - rightLane[0].y ) / avg_slope_r;
        Point target_pos = Point( ( x_right + x_left ) / 2, Y );
        Point center_pos = Point( int( img.cols / 2 ), img.rows );

        // 2. Visualize the target spots
        circle( img_steer, target_pos, 10, Scalar( 0, 255, 0 ), 2 );          /// target circle
        circle( img_steer, center_pos, 10, Scalar( 0, 255, 0 ), 2 );          /// current circle
        circle( img_steer, Point( x_left, Y ), 10, Scalar( 0, 255, 0 ), 2 );  /// left circle
        circle( img_steer, Point( x_right, Y ), 10, Scalar( 0, 255, 0 ),
                2 );  /// right circle
        line( img_steer, Point( x_left, Y ), Point( x_right, Y ), Scalar( 0, 255, 0 ), 1, LINE_AA );
        // 3. Calculate the steering angle and display it
        string cmd = "PERFECT";
        double steer_error_P = target_pos.x - center_pos.x;
        double steer_error_D = steer_error_P - steer_error_P_old;
        double steer_tar =
            STEER_P * steer_error_P + STEER_I * steer_error_I_sum + STEER_D * steer_error_D;
        steer_error_P_old = steer_error_P;
        steer_error_I_sum += steer_error_P;

        if( steer_tar < 0 )
        {
            cmd = "LEFT";
        }
        else if( steer_tar > 0 )
        {
            cmd = "RIGHT";
        }
        /// normalization between -1 and 1
        steer_norm = 2 * ( steer_tar + STEER_MAX ) / ( 2 * STEER_MAX ) - 1;

        // cout << "P = " << STEER_P << " == " << steer_error_P * STEER_P << " || I = " << STEER_I
        //    << " == " << steer_error_I_sum * STEER_I << " || raw = " << steer_norm
        //    << " || norm = " << steer_norm << endl;

        putText( img_steer, cmd + " " + to_string( steer_norm ), Point( 300, 45 ),
                 FONT_HERSHEY_COMPLEX, 1, Scalar( 0, 255, 0 ), 1 );
    }
    //_________________ ACCELERATION _________________/////
    void accelerationControl()
    {
        /// cm/s to km/h
        double vel_curr = ego_velocity_ * ( 36. / 1000. );
        /// adaptive cruise control
        if( dist_curr < DIST_ACT )
        {
            /// stop stopwatch and take time for integral and derivative
            t_dist_stop = chrono::steady_clock::now();
            t_dist_diff =
                chrono::duration_cast<chrono::milliseconds>( t_dist_stop - t_dist_start ).count();

            /// use halfed velocity as target distance, except when below 20kph, then 20m
            double dist_tar = 0.5 * vel_curr;
            if( vel_curr < 20 )
            {
                dist_tar = DIST_TAR_SLOW;
            }

            double dist_error_P = dist_curr - dist_tar;
            double dist_error_D = ( dist_error_P - dist_error_P_old ) / t_dist_diff;

            /// avoid irrational time measurements
            if( t_dist_diff < 100 )
            {
                double dist_error_I_next =
                    ( (double)DIST_I / 1e6 ) * ( dist_error_I_sum + dist_error_P * t_dist_diff );
                /// allow the I part to only influence +- 2kph and include time since last call
                if( dist_error_I_next <= 2 )
                {
                    dist_error_I_sum += dist_error_P * t_dist_diff;
                }
            }

            /// PID controller for target velocity with vel_curr aus offset
            vel_tar = vel_curr + dist_error_P * ( (double)DIST_P / 10 ) +
                      dist_error_I_sum * ( (double)DIST_I / 1e6 ) +
                      dist_error_D * ( (double)DIST_D / 1e6 );

            ///// in braking scenarios, remove the offset again, including hysteresis
            //         if( ( dist_curr < DIST_TAR_SLOW || abs( dist_curr - DIST_TAR_SLOW ) < 3 ) )
            //         {
            //             vel_tar -= vel_curr;
            //         }

            /// avoid negative target velocities
            if( vel_tar < 1 )
            {
                vel_tar = 0;
            }

            /// reduce I when driving slow and getting close to the target vehicle
            if( ( dist_curr < DIST_TAR_SLOW || abs( dist_curr - DIST_TAR_SLOW ) < 3 ) &&
                vel_tar < 5 )
            {
                dist_error_I_sum *= 0.9;
                cout << "dist I reduction" << endl;
            }

            /// no new distance updates from tronis = reset to CC
            dist_curr_old = dist_curr;
            if( dist_curr == dist_curr_old )
            {
                dist_watchdog++;
                // cout << " WD+1: " << dist_watchdog << endl;
                if( dist_watchdog >= DIST_WATCHDOG_MAX )
                {
                    dist_curr = DIST_ACT + 1;
                    dist_watchdog = 0;
                }
            }
            /// start stopwatch
            t_dist_start = chrono::steady_clock::now();

            cout << "distance: " << dist_curr << " m, target: " << dist_tar
                 << " m || P = " << ( (double)DIST_P / 10 )
                 << " == " << dist_error_P * ( (double)DIST_P / 10 )
                 << ", I = " << ( (double)DIST_I / 1e6 )
                 << " == " << dist_error_I_sum * ( (double)DIST_I / 1e6 )
                 << ", D = " << ( (double)DIST_D / 1e6 )
                 << " == " << dist_error_D * ( (double)DIST_D / 1e6 ) << " || cmd: " << vel_tar
                 << " kmh" << endl;

            /// call cruiseControl to apply calculated target velocity
            velocityControl( vel_curr, vel_tar, true );
        }
        else
        {
            /// velocity controller
            velocityControl( vel_curr, vel_tar, false );
        }
    }
    void velocityControl( double vel_c, double vel_t, bool acc_flag )
    {
        /// if acc is active, then limit the target velocity by VEL_TAR (e.g. speed signs)
        if( acc_flag )
        {
            if( VEL_TAR < vel_t )
            {
                vel_t = VEL_TAR;
            }
        }
        else
        {
            vel_t = VEL_TAR;
        }
        /// similar to acceleration PID control
        double vel_error_P = vel_t - vel_c;
        double vel_error_D = ( vel_error_P - vel_error_P_old );
        double acc_tar = ( (double)ACC_P / 1000 ) * vel_error_P +
                         ( (double)ACC_I / 1000000 ) * vel_error_I_sum +
                         ( (double)ACC_D / 1000 ) * vel_error_D;
        vel_error_P_old = vel_error_P;
        /// no multiplication with time, bc controller is stable as is
        vel_error_I_sum += vel_error_P;

        /// below 5m, brake hard
        if( dist_curr < 7.5 )
        {
            acc_tar = -1;
            cout << "hard brake" << endl;
        }

        /// reduce I when driving slow and getting close to the target vehicle
        if( ( dist_curr < DIST_TAR_SLOW || abs( dist_curr - DIST_TAR_SLOW ) < 3 ) && vel_tar < 5 )
        {
            vel_error_I_sum *= 0.;
            cout << "vel I  reduction" << endl;

            /// avoid slow rolling
            if( vel_c < 1 )
            {
                acc_tar = 0;
            }
        }

        /// avoid driving backwards
        if( vel_c < 2 && acc_tar < 0 )
        {
            acc_tar = 0;
            cout << "avoid backwards driving" << endl;
        }

        acc_norm = acc_tar;
        /// hard cut-off at 1 and -1
        if( acc_norm > 1 )
        {
            acc_norm = 1;
        }
        else if( acc_norm < -1 )
        {
            acc_norm = -1;
        }
        cout << "velocity: " << vel_c << " kmh, target: " << vel_t
             << " kmh || P = " << ( (double)ACC_P / 1000 )
             << " == " << vel_error_P * ( (double)ACC_P / 1000 )
             << ", I = " << ( (double)ACC_I / 1000000 )
             << " == " << vel_error_I_sum * ( (double)ACC_I / 1000000 )
             << ", D = " << ( (double)ACC_D / 1000 )
             << " == " << vel_error_D * ( (double)ACC_D / 1000 ) << " || cmd = " << acc_norm
             << endl;
    }
    //_________________ DEFAULT FUNCTIONS _________________/////
public:
    /// Function to process received tronis data
    bool getData( tronis::ModelDataWrapper data_model )
    {
        if( data_model->GetModelType() == tronis::ModelType::Tronis )
        {
            // std::cout << "Id: " << data_model->GetTypeId() << ", Name: " << data_model->GetName()
            //          << ", Time: " << data_model->GetTime() << std::endl;

            // if data is sensor output, process data
            switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
            {
                case tronis::TronisDataType::Image:
                {
                    processImage( data_model->GetName(),
                                  data_model.get_typed<tronis::ImageSub>()->Image );
                    break;
                }
                case tronis::TronisDataType::ImageFrame:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFrameSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::ImageFramePose:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::PoseVelocity:
                {
                    processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                    /* ---------- 6. ACCELERATION CONTROL ---------- */
                    // cout << "6. ACCELERATION" << endl;
                    accelerationControl();
                    break;
                }
                case tronis::TronisDataType::BoxData:
                {
                    // std::cout << data_model->ToString() << std::endl;
                    processBox( data_model.get_typed<tronis::BoxDataSub>() );

                    break;
                }
                default:
                {
                    break;
                }
                    // call
            }
            return true;
        }
        else
        {
            // std::cout << data_model->ToString() << std::endl;
            return false;
        }
    }

protected:
    /// Function to show an openCV image in a separate window
    void showImage( std::string image_name, cv::Mat image )
    {
        cv::Mat out = image;
        if( image.type() == CV_32F || image.type() == CV_64F )
        {
            cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
        }
    }
    /// Function to convert tronis image to openCV image
    bool processImage( const std::string& base_name, const tronis::Image& image )
    {
        // std::cout << "processImage" << std::endl;
        if( image.empty() )
        {
            std::cout << "empty image" << std::endl;
            return false;
        }

        image_name_ = base_name;
        image_orig = tronis::image2Mat( image );

        detectLanes();
        showImage( image_name_, image_orig );

        return true;
    }
    /// Function to convert tronis velocity to processible format
    bool processPoseVelocity( tronis::PoseVelocitySub* msg )
    {
        ego_location_ = msg->Location;
        ego_orientation_ = msg->Orientation;
        ego_velocity_ = msg->Velocity;
        return true;
    }
    /// Function to convert tronis bounding boxes to procissible format
    bool processBox( tronis::BoxDataSub* msg )
    {
        vector<string> box_names;
        vector<double> box_distances;
        /// loop through all detected boxes
        for( int i = 0; i < msg->Objects.size(); i++ )
        {
            // std::cout << msg->ToString() << std::endl;
            tronis::ObjectSub& box = msg->Objects[i];

            /// filter for right object size
            // if( box.BB.Extends.X > 100 && box.BB.Extends.X < 400 && box.BB.Extends.Y > 100 &&
            //    box.BB.Extends.Y < 300 )
            if( box.BB.Extends.X > 100 && box.BB.Extends.X < 800 && box.BB.Extends.Y > 100 &&
                box.BB.Extends.Y < 800 )
            {
                /// remove own vehicle from possibilities
                if( box.Pose.Location.X != 0.0 )
                {
                    /// remove vehicles from parallel lanes
                    if( abs( box.Pose.Location.Y ) < 400 )
                    {
                        // cout << box.ActorName.Value() << ", is " << _hypot( box.Pose.Location.X /
                        // 100, box.Pose.Location.Y / 100 ) << " m ahead." << endl;
                        double dist_curr_temp =
                            _hypot( box.Pose.Location.X / 100, box.Pose.Location.Y / 100 );
                        if( dist_curr_temp > 5 )
                        {
                            /// compensate center position of box sensor: 2.5 (own) + 2.5m
                            /// (preceeding car)
                            dist_curr_temp -= 5;
                        }
                        /// append to vectors
                        box_names.push_back( box.ActorName.Value() );
                        box_distances.push_back( dist_curr_temp );
                    }
                }
            }
        }
        /// find minimum distance box
        double box_min_it = -1;
        double box_min = 100;
        for( int i = 0; i < box_names.size(); i++ )
        {
            // cout << "Box " << i << ": " << box_names[i] << " (" << box_distances[i] << "m)" <<
            // endl;
            if( box_distances[i] < box_min )
            {
                box_min = box_distances[i];
                box_min_it = i;
            }
        }
        /// use min distance box for distance control (in case there are multiple cars)
        if( box_min_it != -1 )
        {
            // cout << "Target Box " << box_min_it << ": " << box_names[box_min_it] << " (" <<
            // box_distances[box_min_it] << "m)" << endl;
            dist_curr = box_distances[box_min_it];
        }

        return true;
    }
};

/// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    // specify socket parameters
    std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "7778";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip
                  << "\", PortBind:" << socket_port << "}";

    int key_press = 0;  // close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500;  // close grabber, if last received msg is older than this param

    LaneAssistant lane_assistant;

    while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );

        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
        tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
            // wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
                // data received! reset timer
                time_ms = 0;

                // convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
                // identify data type
                lane_assistant.getData( data_model );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
                // no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}
