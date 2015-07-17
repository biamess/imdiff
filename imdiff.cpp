/* imdiff.cpp - visual alignment of two images
*
* VS version
* working version as of May 31 2013
* added github control 6/17/2013
* added computation on image pyramids 6/18/2013
* made to compile on Macs and Linux 6/4/2015
* added confidence computation and GT image warping 7/10/15
*/

/* This visual studio project requires two windows environment variables to be set.  Example:
 * OPENCV         D:\opencv-2.4.6
 * OPENCVversion  246
 *
 * based on these variables, it then uses includes and libraries, e.g.:
 * D:\opencv-2.4.6\build\include
 * D:\opencv-2.4.6\build\x86\vc11\lib\opencv_core246.lib
 *
 * In addition, the system path needs to include the following location so that DLLs can be found:
 * D:\opencv-2.4.6\build\x86\vc11\bin
 *
 * *** 2015: changed to use opencv-3.0.0, vc12 -- need to update project settings and these instructions
 */

// set to 1 if running on cygwin - turns off mouse motion animation, o/w crashes on cygwin
int cygwinbug = 0;

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "ImageIOpfm.h"
#include <cmath>			// used for comparing with infinity

#define UNK INFINITY

#if defined(__linux__) || defined(__APPLE__)
#define sprintf_s snprintf
#endif

using namespace cv;
using namespace std;


typedef vector<Mat> Pyr;

Mat oim0, oim1;       // original images
Rect roi;             // cropping rectangle
Mat im0, im1;         // cropped images
Mat gtd;			  // ground truth disparity
Mat occmask;		  // occlusion mask
Pyr pyr0, pyr1, pyrd; // image pyramids of cropped regions

int pyrlevels = 0;  // levels in pyramid (will be determined based on cropped image size)

int mode = 0;
const int nmodes = 4;
const char *modestr[nmodes] = {
	"diff  ",  // color diff 
	"NCC   ",
	"ICPR  ",  // ICPR 94 gradient diff
	"Bleyer"}; // 0.1 * color diff + 0.9 * gradient diff

const char *win = "imdiff";
const char *winConf = "Confidence";
const char *selectedWin;	// currently selected window
float dx = 0;  // offset between images
float dy = 0;
float dgx = 0; // disparity gradient
float ds = 1;  // motion control multiplier
int xonly = 0; // constrain motion in x dir
float startx;
float starty;
float startdy;
float diffscale = 1;
float confScale = 1;
float step = 1;    // arrow key step size
int nccsize = 3;
float ncceps = 1e-2f;
int aggrsize = 1;
int diffmin = 0; // 0 or 128 to clip negative response
int pixshift = 1; // amount (in pixels) to shift the image by

void printhelp()
{
	printf(
		"drag to change offset, shift-drag for fine control\n"
		"control-drag to restrict motion in X only\n"
		"arrows: change offset by stepsize\n"
		"C, V  - change step size\n"
		"O, P  - change disp x gradient\n"
		"Space - reset offset\n"
		"A, S  - show (blink) orig images\n"
		"D     - show diff\n"
		"W 	   - show GT warped image 2\n"
		"+     - show confidence measure\n"
		"Y, U  - confidence contrast\n"
		"1-4 - change mode:\n"
		"  1 - color diff\n"
		"  2 - NCC\n"
		"  3 - ICPR cost\n"
		"  4 - Bleyer cost\n"
		"B     - toggle clipping at 0 (modes 2-4)\n"
		"Z, X  - change diff contrast (mode 1)\n"
		"E, R  - change NCC epsilon (mode 2)\n"
		"N, M  - change NCC window size (mode 2)\n"
		"F, G  - change aggregation window size (modes 2-4)\n"
		"JKLI  - move cropped region (large imgs only)\n"
		"H , ? - help\n"
		"(-)   - close current window\n"
		"Esc, Q - quit\n");
}

void computeGradientX(Mat img, Mat &gx)
{
	int gdepth = CV_32F; // data type of gradient images
	Sobel(img, gx, gdepth, 1, 0, 3, 1, 0);
}

void computeGradientY(Mat img, Mat &gy)
{
	int gdepth = CV_32F; // data type of gradient images
	Sobel(img, gy, gdepth, 0, 1, 3, 1, 0);
}

void computeGradients(Mat img, Mat &gx, Mat &gy, Mat &gm) 
{
	computeGradientX(img, gx);
	computeGradientY(img, gy);
	magnitude(gx, gy, gm);
}

void info(Mat imd)
{
	//rectangle(imd, Point(0, 0), Point(150, 20), Scalar(100, 100, 100), CV_FILLED); // gray rectangle
	//Mat r = imd(Rect(0, imd.rows-18, imd.cols, 18));  // better: darken subregion!
	//r *= 0.5;
	char txt[100];
	char txt3[100];
	if (mode == 0) { // color diff
		sprintf_s(txt, 100, "1-diff * %.1f  dx=%4.1f dy=%4.1f step=%3.1f", 
			diffscale, dx, dy, step);
	} else if (mode == 1) { // NCC
		sprintf_s(txt, 100, "2-NCC %dx%d dx=%4.1f dy=%4.1f step=%3.1f aggr %dx%d ncceps=%5g", 
			nccsize, nccsize, dx, dy, step, aggrsize, aggrsize, ncceps);
	} else {
		sprintf_s(txt, 100, "%d-%s dx=%4.1f dy=%4.1f step=%3.1f aggr %dx%d", 
			mode+1, modestr[mode], dx, dy, step, aggrsize, aggrsize);
	}
	sprintf_s(txt3, 100, "confScale=%.1f", confScale);
	putText(imd, txt, Point(5, imd.rows-15), FONT_HERSHEY_PLAIN, 0.8, Scalar(200, 255, 255));
	const char *txt2 = "C/V:step  O/P:dgx  Z/X:contrast  N/M:nccsize  E/R:ncceps  F/G:aggr  ?:help  Q:quit";
	putText(imd, txt2, Point(5, imd.rows-2), FONT_HERSHEY_PLAIN, 0.8, Scalar(120, 180, 180));
	putText(imd, txt3, Point(400, imd.rows-15), FONT_HERSHEY_PLAIN, 0.8, Scalar(200, 255, 255));
}

void myImDiff2(Mat a, Mat b, Mat &d)
{
	d = 128 + a - b;
}

void myImDiff(Mat a, Mat b, Mat &d)
{
	if (! d.data || d.rows != a.rows || d.cols != a.cols)
		d = a.clone();
	int w = a.cols, h = a.rows, nb = a.channels();

	for (int y=0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			Vec3b pa = a.at<Vec3b>(y, x);
			Vec3b pb = b.at<Vec3b>(y, x);
			Vec3b pd;// = pa - pb;
			for (int z = 0; z < nb; z++) {
				pd[z] = saturate_cast<uchar>(pa[z] - pb[z] + 128);
			}
			d.at<Vec3b>(y, x) = pd;
		}
	}
}

void myImDiff3(Mat a, Mat b, Mat &d)
{
	if (! d.data || d.rows != a.rows || d.cols != a.cols)
		d = a.clone();
	int w = a.cols, h = a.rows, nb = a.channels();

	for (int y=0; y < h; y++) {
		uchar *pa = a.ptr<uchar>(y);
		uchar *pb = b.ptr<uchar>(y);
		uchar *pd = d.ptr<uchar>(y);
		int wnb = w * nb;
		for (int x = 0; x < wnb; x++) {
			pd[x] = saturate_cast<uchar>(pa[x] - pb[x] + 128);
		}
	}
}

void boxFilter(Mat src, Mat &dst, int n) {
	blur(src, dst, Size(n, n), Point(-1, -1));
}

void ncc(Mat L, Mat R, Mat &imd) {
	Mat Lb, Rb;
	boxFilter(L,  Lb, nccsize);
	boxFilter(R,  Rb, nccsize);
	Mat LL = L.mul(L);
	Mat RR = R.mul(R);
	Mat LR = L.mul(R);
	Mat LLb, RRb, LRb;
	boxFilter(LL,  LLb, nccsize);
	boxFilter(RR,  RRb, nccsize);
	boxFilter(LR,  LRb, nccsize);
	Mat LL2 = LLb - Lb.mul(Lb);
	Mat RR2 = RRb - Rb.mul(Rb);
	Mat LR2 = LRb - Lb.mul(Rb);
	Mat den = LL2.mul(RR2) + ncceps;
	sqrt(den, den);
	Mat ncc = LR2 / den;
	ncc.convertTo(imd, CV_8U, 128, 128);
}

// compute image "difference" according to mode
void imdiff(Mat im0, Mat im1, Mat &imd)
{
	if (mode == 0 || mode >= nmodes) { // difference of images
		//addWeighted(im0, diffscale, im1, -diffscale, 128, imd);
		imd = 128 + diffscale * ((im0 - im1)/2);
		return;
	}
	Mat im0g, im1g;
	cvtColor(im0, im0g, CV_BGR2GRAY);  
	cvtColor(im1, im1g, CV_BGR2GRAY);  

	if (mode == 1) { // NCC
		Mat im0gf, im1gf;
		im0g.convertTo(im0gf, CV_32F);
		im1g.convertTo(im1gf, CV_32F);
		ncc(im0gf, im1gf, imd);
	} else if (mode == 2) { // ICPR gradient measure
		Mat gx0, gy0, gx1, gy1, gm0, gm1; // gradients
		computeGradients(im0g, gx0, gy0, gm0);
		computeGradients(im1g, gx1, gy1, gm1);
		gm1 += gm0; // sum of the gradient magnitudes s
		gx1 -= gx0; // compute magnitude of difference
		gy1 -= gy0;
		Mat gmag;
		magnitude(gx1, gy1, gmag); // magnitude of difference d
		addWeighted(gm1, 0.5, gmag, -1, 128, imd, CV_8U); // result is s/2 - d
	} else 	if (mode == 3) { // Bleyer weighted sum of color and gradient diff
		Mat cdiff, gx0, gx1, gdiff;
		absdiff(im0, im1, cdiff);
		cvtColor(cdiff, cdiff, CV_BGR2GRAY);
		computeGradientX(im0g, gx0);
		computeGradientX(im1g, gx1);
		absdiff(gx0, gx1, gdiff);
		gdiff.convertTo(imd, CV_8U, 1, 0); // only show x-grad diff right now
		//float sc = diffscale;
		//addWeighted(cdiff, 0.1*sc, gdiff, 0.9*sc, 0, imd, CV_8U);
		//imd = 255 - imd;
		//still need to truncate diffs

	}
	imd = max(imd, diffmin);
	if (aggrsize > 1)
		boxFilter(imd, imd, aggrsize);
}

Mat pyrImg(Pyr pyr) 
{
	Mat im = pyr[0];
	int w = im.cols, h = im.rows;
	Mat pim(Size(3*w/2+4, h+30), im.type(), Scalar(30, 10, 10));
	im.copyTo(pim(Rect(0, 0, w, h)));

	int x = w+2;
	int y = 0;
	for (int i = 1; i < (int)pyr.size(); i++) {
		int w1 = pyr[i].cols, h1 = pyr[i].rows;
		pyr[i].copyTo(pim(Rect(x, y, w1, h1)));
		y += h1 + 2;
	}
	return pim;
}

// print information about image
void printinfo(Mat img)
{
	printf("width=%d, height=%d, channels=%d, pixel type: ",
		img.cols, img.rows, img.channels());
	switch(img.depth()) {
	case CV_8U:	 printf("CV_8U  -  8-bit unsigned int (0..255)\n"); break;
	case CV_8S:  printf("CV_8S  -  8-bit signed int (-128..127)\n"); break;
	case CV_16U: printf("CV_16U - 16-bit unsigned int (0..65535)\n"); break;
	case CV_16S: printf("CV_16S - 16-bit signed int (-32768..32767)\n"); break;
	case CV_32S: printf("CV_32S - 32-bit signed int\n"); break;
	case CV_32F: printf("CV_32F - 32-bit float\n"); break;
	case CV_64F: printf("CV_64F - 64-bit double\n"); break;
	default:     printf("unknown\n");
	}
}

void dispPyr(const char *window, Pyr pim)
{
	Mat im = pyrImg(pim);
	//im.convertTo(im, CV_8U);
	if (im.channels() != 3)
		cvtColor(im, im, CV_GRAY2BGR);
	info(im);
	imshow(window, im);
}

// move rect within bounds w, h, return whether moved
bool offsetRect(Rect &r, int ddx, int ddy, int w, int h)
{
	ddx = min(w - (r.x + r.width),  max(-r.x, ddx));
	ddy = min(h - (r.y + r.height), max(-r.y, ddy));
	r += Point(ddx, ddy);
	return ddx != 0 || ddy != 0;
}

void imdiff()
{
	// added by Matt Stanley and Bianca Messner to reset im0 pyramid
	// 2015-07-16
	buildPyramid(im0, pyr0, pyrlevels);


	// try to accommodate offset dx, dy by shifting roi in im1
	Rect roi1 = roi;	
	offsetRect(roi1, (int)floor(-dx), (int)floor(-dy), oim1.cols, oim1.rows);
	im1 = oim1(roi1);
	buildPyramid(im1, pyr1, pyrlevels);

	// now compute remaining dx, dy given difference of roi's
	Point dd = roi.tl() - roi1.tl();
	float rdx = dx - dd.x - dgx*im0.rows/2; // rotate around center
	float rdy = dy - dd.y;
	float s = 1;
	//Mat T0 = (Mat_<float>(2,3) << s, 0,  0, 0, s,  0); 
	Mat T1 = (Mat_<float>(2,3) << s, dgx, rdx, 0, s, rdy); 
	//Mat im0t;
	//warpAffine(im0, im0t, T0, im0.size());
	Mat im1t;
	warpAffine(im1, im1t, T1, im1.size());
	buildPyramid(im1t, pyr1, pyrlevels);
	
	for (int i = 0; i <= pyrlevels; i++) {
		imdiff(pyr0[i], pyr1[i], pyrd[i]);
	}
	//printinfo(pyrd[0]);
	dispPyr(win, pyrd);
	
}

// Added passing window name through param to get currently selected
// window, so the user can close the selected window - Matt Stanley 7/14/2015
static void onMouse( int event, int x, int y, int flags, void *param)
{
	const char* winID = (const char*)param;
	if(winID == win){
		selectedWin = win;
	}else if(winID == winConf){
		selectedWin = winConf;
		return;
	}

	x = (short)x; // seem to be short values passed in, cast needed for negative values during dragging
	y = (short)y;
	//printf("x=%d y=%d    ", x, y);
	//printf("dx=%g dy=%g\n", dx, dy);
	if (event == CV_EVENT_LBUTTONDOWN) {
		ds = (float)((flags & CV_EVENT_FLAG_SHIFTKEY) ? 0.1 : 1.0); // fine motion control if Shift is down
		xonly = flags & CV_EVENT_FLAG_CTRLKEY;  // xonly motion if Control is down
		startx = ds*x - dx;
		starty = ds*y - dy;
		startdy = dy;
		//} else if (event == CV_EVENT_LBUTTONUP) {
		//	imdiff();
	} else if (event == CV_EVENT_MOUSEMOVE && flags & CV_EVENT_FLAG_LBUTTON) {
		xonly = flags & CV_EVENT_FLAG_CTRLKEY;  // xonly motion if Control is down
		dx = ds*x - startx;
		if (!xonly)
			dy = ds*y - starty;
		else
			dy = startdy;
		if (!cygwinbug)
			imdiff();
	}
}


void shiftROI(int ddx, int ddy) {
	if (offsetRect(roi, ddx, ddy, oim0.cols, oim0.rows)) {
		im0 = oim0(roi);
		buildPyramid(im0, pyr0, pyrlevels);
		imdiff();
	}
}


//converts a 3-band image "src" into a single band image stored at "dst"
//by summing up the values in the three channels
void sumChannels (Mat src, Mat &dst){
	//START Creating 1-channel images
		
	//init images to hold summed channels to be 
	//correct width, height and type
	int height = src.size().height;
	int width = src.size().width;
	int type = CV_32FC1;
	dst = Mat::zeros(height, width, type);

	//init arrays to hold the 3 images corresponding to each channel
	//of the original
	Mat srcSplit[3];
		

	//split the images into 3 1 channel images
	split(src, srcSplit);


	//sum the individual channels up
	for(int i=0; i < 3; ++i){
		Mat srcF;
			
		//convert CV_8U to CV_32F to avoid overflow
		srcSplit[i].convertTo(srcF, type);

		dst += srcF;
	}
}


//added by Matt Stanley & Bianca Messner 7/10/2015
//compute and display visualization of confidence measure
void computeConf()
{

	namedWindow(winConf, CV_WINDOW_AUTOSIZE);
	setMouseCallback(winConf, onMouse, (void*)winConf);

	Mat im1Copy, im1LShift, im1RShift, RDiff, LDiff, CMMat, CPMat;
	Pyr pyrR, pyrL, pyrConf;
	im1Copy = pyr1[0].clone();

	//initialize pyramids to correct size:
	pyrConf.resize(pyrlevels+1);

	//create matrices to shift the image by 1 pixshift to the right and left
	Mat LShift = (Mat_<float>(2,3) << 1, 0, -pixshift, 0, 1, 0);
	Mat RShift = (Mat_<float>(2,3) << 1, 0, pixshift, 0, 1, 0);

	//shift the image to the right and left
	warpAffine(im1Copy, im1LShift, LShift, im1Copy.size());
	warpAffine(im1Copy, im1RShift, RShift, im1Copy.size());

	//create image pyramids from the shifted images
	buildPyramid(im1LShift, pyrL, pyrlevels);
	buildPyramid(im1RShift, pyrR, pyrlevels);

	for (int i = 0; i <= pyrlevels; i++) {

		//compute the matching costs between the images in the shifted pyramids
	    //and the bottom image
		imdiff(pyr0[i], pyrL[i], LDiff);
		imdiff(pyr0[i], pyrR[i], RDiff);

		//corrected version of matching cost images where 0 indicates perfect match (rather than 128)
		Mat corrDC3 = abs(pyrd[i] - 128);
		Mat corrLDiffC3 = abs(LDiff - 128);
		Mat corrRDiffC3 = abs(RDiff - 128);

		//compute absolute differences for 3 channel image
		absdiff(corrDC3, corrLDiffC3, CMMat);
		absdiff(corrDC3, corrRDiffC3, CPMat);


		//create 1-channel images:
		//init images to hold summed channels
		Mat sumChannelsCMS;
		Mat sumChannelsCPS;
		Mat sumChannelsdS;
		Mat sumChannelsLS;
		Mat sumChannelsRS;

		sumChannels(CMMat, sumChannelsCMS);
		sumChannels(CPMat, sumChannelsCPS);
		sumChannels(corrDC3, sumChannelsdS);
		sumChannels(LDiff, sumChannelsLS);
		sumChannels(RDiff, sumChannelsRS);


		//identify the pixels for which the current disparity yields a local minimum in matching costs
		//create masks for the left and right images so that we can have 0 confidence
		//at the pixels for which the current disparity is not a local minimum
		Mat LMask = (sumChannelsdS < sumChannelsLS); 	//mask for the L image - 255's where the d image holds the min,
									  	  				//zeros elsewhere
		Mat RMask = (sumChannelsdS < sumChannelsRS); 	//mask for the R image - 255's where the d image holds the min,
									      				//zeros elsewhere

		//scale the values in the masks so that the 255's become 1's
		LMask /= 255.0;
		RMask /= 255.0;

		LMask.convertTo(LMask, CV_32FC1);
		RMask.convertTo(RMask, CV_32FC1);

		//mask out the values where the current disparity was not the minimum
		//these will now hold zeroes, which will always be the minimum since the 
		//absdiff is always positive
		sumChannelsCMS = LMask.mul(sumChannelsCMS); 
		sumChannelsCPS = RMask.mul(sumChannelsCPS);

		//identify the min of the absolute differences computed above and
		//store that in a new confidence pyramid
		Mat CMMask = (sumChannelsCMS < sumChannelsCPS); //mask for the CM image - 255's where the CM (left) image holds the min,
											//zeros elsewhere
		Mat CPMask = (sumChannelsCPS <= sumChannelsCMS); //mask for the CP image - 255's where the CP (right) image holds the min
		//scale the values in the masks so that the 255's become 1's

		CMMask /= 255.0;
		CPMask /= 255.0;

		CMMask.convertTo(CMMask, sumChannelsCMS.type());
		CPMask.convertTo(CPMask, sumChannelsCPS.type());

		//do element-wise multiplication between the masks and the absdiff matrices
		//to produce matrices that hold the minimum value where the mask had a 1 and zeroes elsewhere
		Mat minCM = CMMask.mul(sumChannelsCMS); 
		Mat minCP = CPMask.mul(sumChannelsCPS);

		//add the matrices computed above so that we have the min values at each cell
		//(where one matrix held a min value, the other will hold a zero since
		//when computing the masks, either pyrCM[i] < pyrCP[i] or pyrCP[i] < pyrCM[i],
		//not both)
		pyrConf[i] = ((minCM + minCP)/3) * confScale;
	}	

	//display the confidence measure in a new window
	dispPyr(winConf, pyrConf);
}


// bilinear interpolation of ints in 0..255
// taken from Daniel's warp.cpp
int linearInterpi(float fx, float fy, int v00, int v01, int v10, int v11)
{
	float w00 = (1-fx)*(1-fy);
	float w01 = (1-fx)*fy;
	float w10 = fx*(1-fy);
	float w11 = fx*fy;

	float v = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
	int vr = round(v);
	return max(0, min(255, vr));
}


// added by Matt Stanley & Bianca Messner 2015-07-15
// warps an image by it's ground truth disparities
void warpImageInv(Mat src, Mat &dst, Mat dispx, float scalex=1.0)
{
	// get dimensions and type of src image
	int width = src.size().width, height = src.size().height;
	int type = src.type();	
	int nB = src.channels();

	// make sure src image is a color image									
	if(nB != 3)
	{
		cout << "expected color image" << endl;
		return;
	}

	// initialize the dst image to be bright pink
	dst = Mat(height, width, type, Scalar(255, 128, 255));

	
	int n = 0;
	// begin pixel loop
	for (int y = 0; y < height; ++y)	// rows
	{
		for (int x = 0; x < width; ++x)	// cols
		{
			float dx = scalex * dispx.at<float>(y, x); // index into dispx using (row, col)=(y, x)

			if(dx == UNK)
				continue;
			n++;

			float xx = x + dx;
			float yy = y;

			int ixr = (int)round(xx);
			int iyr = (int)round(yy);

			if(ixr < 0 || ixr >= width || iyr < 0 || iyr >= height)
				continue;

			int ix0 = max(0, (int)floor(xx));
			int iy0 = max(0, (int)floor(yy));
			int ix1 = min(width-1, ix0+1);
			int iy1 = min(height-1, iy0+1);

			float fx = xx - ix0;
			float fy = yy - iy0;

			for(int b = 0; b < nB; ++b){
				dst.at<Vec3b>(y,x)[b] = linearInterpi(	fx, fy,
														src.at<Vec3b>(iy0, ix0)[b],
														src.at<Vec3b>(iy1, ix0)[b],
														src.at<Vec3b>(iy0, ix1)[b],
														src.at<Vec3b>(iy1, ix1)[b]
														);
			}
		}
	}
	// end pixel loop
}


// added by Matt Stanley & Bianca Messner 2015-07-15
// warps an image by it's ground truth disparities
// uses opencv's remap() function
void warpImageRemap(Mat src, Mat &dst, Mat dispx, float scalex=1.0)
{
	// get dimensions and type of src image
	int width = src.size().width, height = src.size().height;
	int type = src.type();	
	int nB = src.channels();

	// make sure src image is a color image									
	if(nB != 3)
	{
		cout << "expected color image" << endl;
		return;
	}

	Mat map = Mat_<Point_<float> >(height, width);
	// initialize the dst image to be bright pink
	dst = Mat(height, width, type, Scalar(255, 128, 255));
	Mat emptyMap;

	float gtValue;

	// begin building map
	for(int i=0; i<height; ++i)
	{
		for (int j=0; j < width; ++j)
		{
			gtValue = scalex * (dispx.at<float>(i, j));
			if (!((j+gtValue) > width)){ // if not out of bounds...
				map.at<Point_<float> >(i, j) = Point(j+gtValue, i); //store the coordinates of the point in the 
														   			//source image that contains the pixel we want in
														   			//the destination image
			}
		}
	}
	// end building map
	remap(src, dst, map, emptyMap, INTER_LINEAR);
}


// added by Bianca Messner & Matt Stanley 2015-07-16
// given an occlusion mask sets the corresponding pixels
// in the src image to be the color (0, 255, 0)
void maskOccluded(Mat src, Mat &dst, Mat occlusionmask) {
	// get src image dimensions
	int width = src.size().width, height = src.size().height;
	int type = src.type();	

	// create a mask for non-occluded areas and
	// occluded areas (both half and full)
	Mat nonOccMask = (occlusionmask == 255)/255;
	Mat occAreaMask = (occlusionmask != 255)/255;

	// cut out colored pieces in the occluded areas
	Mat colorMat = Mat(height, width, type, Scalar(0, 255, 0));
	Mat colorOccArea = occAreaMask.mul(colorMat);

	// cut out the color values in the src image
	// that are occluded
	Mat srcNonOcc = src.mul(nonOccMask);

	// fill in occluded areas with color
	dst = srcNonOcc + colorOccArea;
}

//added by Matt Stanley & Bianca Messner 7/10/2015
void warpByGT()
{
	if(gtd.empty()){
		cout << "No Ground Truth image specified" << endl;
		return;
	}

	Mat warped, warped1, warped2, diff;
	warpImageInv(oim1, warped1, gtd, -1);
	warpImageRemap(oim1, warped2, gtd, -1);
	maskOccluded(warped1, warped1, occmask);

	warped1 = warped1(roi);
	warped2 = warped2(roi);

	/*
	imdiff(warped1, warped2, diff);
	imshow("diff", diff);
	imwrite("warpFunctionDiff.png", diff);
	imwrite("warpRemap.png", warped2);
	imwrite("warpOurs.png", warped1);
	*/

	Mat im0MaskOcc;
	maskOccluded(im0, im0MaskOcc, occmask(roi));
	buildPyramid(warped1, pyr1, pyrlevels);
	buildPyramid(im0MaskOcc, pyr0, pyrlevels);

	
	Pyr pyrW1;
	pyrW1.resize(pyrlevels+1);
	buildPyramid(warped1, pyrW1, pyrlevels);
	for(int i=0; i<pyrlevels; ++i){
		imdiff(pyr0[i], pyrW1[i], pyrd[i]);
	}

	//Mat maskedDiff;
	//maskOccluded(pyrd[0], maskedDiff, occmask(roi));
	//buildPyramid(maskedDiff, pyrd, pyrlevels);
	dispPyr("warped", pyrd);
	

	computeConf();
}


void mainLoop()
{
	Mat tmp;
	Mat dst;

	while (1) {
		int c = waitKey(0);
		switch(c) {
		case 27: // ESC
		case 'q':
			return;
		case 7602176: // F5
			{
				//Mat m1 = imd;
				break;	// can set a breakpoint here, and then use F5 to stop and restart
			}
		case 'h':
		case '?':
			printhelp(); break;
		case 45:
			destroyWindow(selectedWin); break;
		case 2424832: case 65361: // left arrow
			dx -= step; imdiff(); break;
		case 2555904: case 65363: // right arrow
			dx += step; imdiff(); break;
		case 2490368: case 65362: // up arrow
			dy -= step; imdiff(); break; 
		case 2621440: case 65364: // down arrow
			dy += step; imdiff(); break;
		case 'c': // decrease step
			step /= 2; imdiff(); break;
		case 'v': // increase step
			step *= 2; imdiff(); break;
		case 'o': // increase x disp gradient
			dgx += 0.02f; imdiff(); break;
		case 'p': // decrease x disp gradient
			dgx -= 0.02f; imdiff(); break;
		case ' ': // reset
			dx = 0; dy = 0; dgx = 0; diffscale = 1; nccsize = 3; imdiff(); break;
		case 'a': // show original left image
			dispPyr(win, pyr0); break;
		case 's': // show transformed right image
			dispPyr(win, pyr1); break;
		case 'd': // back to diff
			imdiff(); break;
		case '+': // display confidence image
			computeConf(); break;
		case 'w':
			warpByGT(); break;
		case 'y':
			confScale += 0.10; imdiff(); break;
		case 'u':
			confScale = max(0.0, confScale-0.10); imdiff(); break;
		case 'z': // decrease contrast
			if (mode==0) {diffscale /= 1.5; imdiff();} break;
		case 'x': // increase contrast
			if (mode==0) {diffscale *= 1.5; imdiff();} break;
		case 'e': // decrease eps
			if (mode==1) {ncceps /= 2; imdiff();} break;
		case 'r': // increase eps
			if (mode==1) {ncceps *= 2; imdiff();} break;
		case 'n': // decrease NCC window size
			if (mode==1) {nccsize = max(nccsize-2, 3); imdiff();} break;
		case 'm': // increase NCC window size
			if (mode==1) {nccsize += 2; imdiff();} break;
		case 'f': // decrease aggregation window size
			if (mode>0) {aggrsize = max(aggrsize-2, 1); imdiff();} break;
		case 'g': // increase aggregation window size
			if (mode>0) {aggrsize += 2; imdiff();} break;
		case 'b': // toggle clipping negative response
			if (mode>0) {diffmin = 128 - diffmin; imdiff();} break;
		case 'j': // move ROI left
			shiftROI(-30, 0); break;
		case 'l': // move ROI right
			shiftROI(30, 0); break;
		case 'k': // move ROI down
			shiftROI(0, 30); break;
		case 'i': // move ROI up
			shiftROI(0, -30); break;
		case '1': case '2': case '3': case '4':  // change mode
		case '5': case '6': case '7': case '8': case '9':
			mode = min(c - '1', nmodes-1);
			//printf("using mode %s\n", modestr[mode]);
			imdiff(); break;
		case -1: // happens when the user closes the window
			return;
		default:
			printf("key %d (%c %d) pressed\n", c, (char)c, (char)c);
		}
	}
}

Mat readIm(char *fname) 
{
	Mat im = imread(fname, 1);
	if (im.empty()) { 
		fprintf(stderr, "cannot read image %s\n", fname); 
		exit(1); 
	}
	return im;
}

void ensureSameSize(Mat &im0, Mat &im1, bool exitIfDifferent=false) 
{
	if (im0.rows == im1.rows && im0.cols == im1.cols)
		return;
	if (exitIfDifferent) {
		fprintf(stderr, "images must have the same size\n"); 
		exit(1); 
	}
	// otherwise crop to common size
	Rect r(Point(0, 0), im0.size());
	r &= Rect(Point(0, 0), im1.size());  //Rectangle intersection
	im0 = im0(r);
	im1 = im1(r);
}

int main(int argc, char ** argv)
{
	setvbuf(stdout, (char*)NULL, _IONBF, 0); // fix to flush stdout when called from cygwin

	if (argc < 3) {
		fprintf(stderr, "usage: %s im1 im2 [groundtruthim2 [occlusionmask [decimationFact [offsx [offsy]]]]\n", argv[0]);
		exit(1);
	}
	try {
		oim0 = readIm(argv[1]);
		oim1 = readIm(argv[2]);
		ensureSameSize(oim0, oim1);


		int downsample = 1; // use 2 for half-size, etc.
		int offsx = -1, offsy = -1;
		if (argc > 3)
			ReadFilePFM(gtd, argv[3]);
		if (argc > 4)
			occmask = readIm(argv[4]);
		if (argc > 5)
			downsample = atoi(argv[5]);
		if (argc > 6)
			offsx = atoi(argv[6]);
		if (argc > 7)
			offsy = atoi(argv[7]);

		// downsample images
		if (downsample > 1) {
			double f = 1.0/downsample;
			resize(oim0, oim0, Size(), f, f, INTER_AREA);
			resize(oim1, oim1, Size(), f, f, INTER_AREA);
		}

		// crop region in images if too big
		int maxw = 600;
		if (oim0.cols > maxw) { // crop subregion
			int w = maxw, h = min(oim0.rows, maxw);
			offsx = (offsx < 0) ? (oim0.cols - w)/2 : max(0, min(oim0.cols - w - 1, offsx));
			offsy = (offsy < 0) ? (oim0.rows - h)/2 : max(0, min(oim0.rows - h - 1, offsy));
			roi = Rect(offsx, offsy, w, h);
			printf("cropping region %dx%d +%d +%d\n", w, h, offsx, offsy);
		} else {
			roi = Rect(0, 0, oim0.cols, oim0.rows);
		}
		im0 = oim0(roi);
		im1 = oim1(roi);

		// determine number of levels in the pyramid
		int smallestpyr = 20; // size of smallest pyramid image
		int minsize = min(im0.rows, im0.cols)/2;
		pyrlevels = 0;
		while (minsize >= smallestpyr) {
			minsize /= 2;
			pyrlevels++;
		}

		buildPyramid(im0, pyr0, pyrlevels);
		buildPyramid(im1, pyr1, pyrlevels);
		pyrd.resize(pyrlevels+1);
		//Mat im = pyr0[pyrlevels];
		//printf("%d levels, smallest size = %d x %d\n", pyrlevels, im.cols, im.rows);

		namedWindow(win, CV_WINDOW_AUTOSIZE);
		setMouseCallback(win, onMouse, (void*)win);
		selectedWin = win;
		imdiff();

		mainLoop();
		destroyAllWindows();
	}
	catch (Exception &e) {
		fprintf(stderr, "exception caught: %s\n", e.what());
		exit(1);
	}
	return 0;
}
