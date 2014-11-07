#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

class ColorTranfer
{
private:
	Mat srcImg;
	Mat targetImg;
	Mat srcImg_lab;//CV_32FC3
	Mat targetImg_lab;//CV_32FC3
	//Mat result_lab;
	vector<float> srcMeans;
	vector<float> srcVariances;
	vector<float> targetMeans;
	vector<float> targetVariances;
	void initialize()
	{
		Mat srcImg_32F, targetImg_32F;
		srcImg.convertTo(srcImg_32F, CV_32FC3, 1.0f / 255.f);
		targetImg.convertTo(targetImg_32F, CV_32FC3, 1.0f / 255.0f);
		cvtColor(srcImg_32F, srcImg_lab, CV_BGR2Lab);
		cvtColor(targetImg_32F, targetImg_lab, CV_BGR2Lab);
		//if(srcImg_lab.depth() == CV_32FC3) cout << "Yes" << endl;
		cout << srcImg_lab.depth() << endl;
		cout << srcImg.depth() << endl;
		computeMeans();
		computeVariances();

	}
	void computeMeans(){
		int srcPixels = srcImg.rows * srcImg.cols;
		int targetPixels = targetImg.rows * targetImg.cols;
		//computing the mean of source image in lab space
		float sum[3] = { 0 };
		for (int row = 0; row < srcImg.rows; row++)
		{
			for (int col = 0; col < srcImg.cols; col++)
			{
				//Point pt(col, row);
				Vec3f color = srcImg_lab.at<Vec3f>(row, col);
				for (int layer = 0; layer < 3; layer++)
					sum[layer] += color[layer];
			}
		}
		for (int i = 0; i < 3; i++)
			srcMeans[i] = sum[i] / srcPixels;

		//computing the mean of target image int lab color space
		for (int i = 0; i < 3; i++)
			sum[i] = 0;
		for (int row = 0; row < targetImg.rows; row++)
		{
			for (int col = 0; col < targetImg.cols; col++)
			{
				//Point pt(col, row);
				Vec3f color = targetImg_lab.at<Vec3f>(row, col);
				for (int layer = 0; layer < 3; layer++)
					sum[layer] += color[layer];
			}
		}
		for (int i = 0; i < 3; i++)
			targetMeans[i] = sum[i] / targetPixels;
	}//end function compuetMeans
	void computeVariances(){
		int srcPixels = srcImg_lab.cols * srcImg_lab.rows;
		int targetPixels = targetImg_lab.cols * targetImg_lab.rows;
		//computing the variance of source image 
		float sum_variance[3] = { 0.f };
		for (int y = 0; y < srcImg_lab.rows; y++)
		{
			for (int x = 0; x < srcImg_lab.cols; x++)
			{
				Vec3f color = srcImg_lab.at<Vec3f>(y, x);
				sum_variance[0] += (color[0] - srcMeans[0])*(color[0] - srcMeans[0]);
				sum_variance[1] += (color[1] - srcMeans[1])*(color[1] - srcMeans[1]);
				sum_variance[2] += (color[2] - srcMeans[2])*(color[2] - srcMeans[2]);
			}
		}
		srcVariances[0] = sqrt(sum_variance[0] / srcPixels);
		srcVariances[1] = sqrt(sum_variance[1] / srcPixels);
		srcVariances[2] = sqrt(sum_variance[2] / srcPixels);

		//computing the variance of target image 
		for (int i = 0; i < 3; i++)
			sum_variance[i] = 0.f;

		for (int y = 0; y < targetImg_lab.rows; y++)
		{
			for (int x = 0; x < targetImg_lab.cols; x++)
			{
				Vec3f color = targetImg_lab.at<Vec3f>(y, x);
				sum_variance[0] += (color[0] - targetMeans[0])*(color[0] - targetMeans[0]);
				sum_variance[1] += (color[1] - targetMeans[1])*(color[1] - targetMeans[1]);
				sum_variance[2] += (color[2] - targetMeans[2])*(color[2] - targetMeans[2]);
			}
		}
		targetVariances[0] = sqrt(sum_variance[0] / targetPixels);
		targetVariances[1] = sqrt(sum_variance[1] / targetPixels);
		targetVariances[2] = sqrt(sum_variance[2] / targetPixels);


	}
public:
	Mat result;
	void solve()
	{
		initialize();
		//constructing the final value in very pixel and store the value in result matrix
		int width = srcImg.cols;
		int height = srcImg.rows;
		//Mat result(height, width, CV_32FC3);
		Mat result_lab(height, width, CV_32FC3);
		float deta_rate[3];
		for (int k = 0; k < 3; k++)
		{
			deta_rate[k] = targetVariances[k] / srcVariances[k];
		}
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				Vec3f color = srcImg_lab.at<Vec3f>(y, x);
				Vec3f value;
				for (int channel = 0; channel < 3; channel++)
					value[channel] = deta_rate[channel] * (color[channel] - srcMeans[channel]) + targetMeans[channel];
				result_lab.at<Vec3f>(y, x) = value;

			}
		}
		//construct the final image
		cvtColor(result_lab, result, CV_Lab2BGR);
	}

	ColorTranfer(Mat src, Mat target) :srcImg(src), targetImg(target)
	{
		srcMeans.resize(3, 0.f);
		srcVariances.resize(3, 0.f);
		targetMeans.resize(3, 0.f);
		targetVariances.resize(3, 0.f);
		solve();
	}
};

int main()
{
	Mat src = imread("11.jpg");
	Mat target = imread("12.jpg");
	namedWindow("src");
	imshow("src", src);
	namedWindow("target");
	imshow("target", target);
	ColorTranfer clt(src, target);
	namedWindow("result");
	imshow("result", clt.result);
	waitKey(0);
	return 0;
}
