/*
  Bruce A. Maxwell
  January 2025

  A simple wrapper for a Depth Anything V2 Network loaded and run
  using the ONNX Runtime API.  It's currently set up to use the CPU.

  When creating a DA2Network object, pass in the path to the Depth
  Anything network.  If using the included model_fp16.onnx method, use
  the constructor with just the path.

  If you want to use a different DA network, use the other
  constructor, which also requires passing the names of the input
  layer and the output layer.  You can find out these values by
  loading the network into the Netron web-app (netron.app).

  This wrapper is intended for a DA network with dynamic sizing.  It
  seems to work best if you use an image of at least 200x200.  Smaller
  images give pretty approximate results.

  The class handles resizing and normalizing the input image with the set_input function.

  The function run_network applies the current input image to the
  network. The result is resized back to the specified image size.
  The result image is a greyscale image with value sin the range of
  [0..255] with 0 being the minimum depth and 255 being the maximum
  depth.  These are not metric values but are scaled relative to the
  network output.

*/
#include <cstdio>
#include <cstring>
#include <cmath>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class DA2Network {
public:

  // constructor with just the network pathname, layer names are hard-coded
  DA2Network( const char *network_path ) {
    std::strncpy( network_path_, network_path, 255 );
    std::strncpy( input_names_, "pixel_values", 255 ); // default values for the network mode_fp16.onnx
    std::strncpy( output_names_, "predicted_depth", 255 );

    // set up the Ort session
    this->session_ = new Ort::Session(env, network_path, Ort::SessionOptions{nullptr});
  }

  // constructor with both the network path and the layer names
  DA2Network( const char *network_path, const char *input_layer_name, const char *output_layer_name ) {
    std::strncpy( network_path_, network_path, 255 );
    std::strncpy( input_names_, input_layer_name, 255 );
    std::strncpy( output_names_, output_layer_name, 255 );

    // set up the Ort session
    this->session_ = new Ort::Session(env, network_path, Ort::SessionOptions{nullptr});
  }

  // deconstructor
  ~DA2Network() {
    if(this->input_data != NULL) { delete[] this->input_data; }
    delete this->session_;
  }

  // accessors
  int in_height(void) {
    return this->height_;
  }

  int in_width(void) {
    return this->width_;
  }

  int out_height(void) {
    return this->out_height_;
  }

  int out_width(void) {
    return this->out_width_;
  }

  // Given a regular image read using cv::imread
  // Rescales and normalizes the image data appropriate for the network
  // scale_factor lets the user resize the image for application to the network
  // smaller images are faster to process, images smaller than 200x200 don't work as well
  int set_input( const cv::Mat &src, const float scale_factor = 1.0 ) {
    cv::Mat tmp;

    // check if we need to resize the input image before applying it to the network
    if( scale_factor != 1.0 ) {
      cv::resize( src, tmp, cv::Size(), scale_factor, scale_factor );
    }
    else {
      tmp = src;
    }

    // check if we need to allocate memory for the input tensor
    if( tmp.rows != this->height_ || tmp.cols != this->width_ ) {
      this->height_ = tmp.rows; // tmp is the image being applied to the network
      this->width_ = tmp.cols;

      if(this->input_data != NULL) {
	delete[] this->input_data;
      }
      
      // allocate the image data
      this->input_data = new float[this->height_ * this->width_ * 3];
      this->input_shape_[2] = this->height_;
      this->input_shape_[3] = this->width_;

      // make the input tensor using the data
      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

      // if using cuda, this is one thing that needs to change
      // auto cuda_mem_info = Ort::MemoryInfo( "cuda", OrtArenaAllocator, OrtMemTypeDefault);
      this->input_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
							    this->input_data,
							    this->height_ * this->width_ * 3,
							    this->input_shape_.data(),
							    this->input_shape_.size());
    }

    // copy the data over to the input tensor data
    // remember, the input data uses a plane representation per color channel, not interleaved
    const int image_size = this->height_ * this->width_;
    for(int i=0;i<tmp.rows;i++) {
      cv::Vec3b *ptr = tmp.ptr<cv::Vec3b>(i);
      float *fptrR = &(this->input_data[i*this->width_]);
      float *fptrG = &(this->input_data[image_size + i*this->width_]);
      float *fptrB = &(this->input_data[image_size*2 + i*this->width_]);
      for(int j=0;j<tmp.cols;j++) {
	fptrR[j] = ((ptr[j][2]/255.0) - 0.485) / 0.229;
	fptrG[j] = ((ptr[j][1]/255.0) - 0.456) / 0.224;
	fptrB[j] = ((ptr[j][0]/255.0) - 0.406) / 0.225;
      }
    }

    // all set to run
    return(0);
  }

  int run_network( cv::Mat &dst, const cv::Size &output_size ) {

    if(this->height_ == 1 || this->width_ == 1) {
      std::cout << "Input tensor not set up, Terminating" << std::endl;
      exit(-1);
    }

    // input_tensor is already set up in set_input
    Ort::RunOptions run_options;

    // run the network, it will dynamically allocate the necessary output memory
    const char* input_names[] = { input_names_ };
    const char* output_names[] = { output_names_ };
    auto outputTensor = session_->Run(run_options, input_names, &input_tensor_, 1, output_names, 1);

    // get the output data size (not quite the same as the input size)
    auto outputInfo = outputTensor[0].GetTensorTypeAndShapeInfo();
    this->out_height_ = outputInfo.GetShape()[1];
    this->out_width_ = outputInfo.GetShape()[2];

    // get the output data
    const float *tensorData = outputTensor[0].GetTensorData<float>();
    static cv::Mat tmp( out_height_, out_width_, CV_8UC1 ); // might as well re-use it if possible

    // get the min and max of the output tensor and copy to a temporary cv::Mat
    float max = -1e+6;
    float min = 1e+6;
    for(int i=0;i<out_height_*out_width_;i++) {
      const float value = tensorData[i];
      min = value < min ? value : min;
      max = value > max ? value : max;
    }

    // copy the normalized data over to a temporary cv::Mat
    // note that there is a little bit of a shift of the depth data to the right
    for(int i=0,k=0;i<out_height_;i++) {
      unsigned char *ptr = tmp.ptr<unsigned char>(i);
      for(int j=0;j<out_width_;j++, k++) {
	float value = 255 * (tensorData[k] - min) / (max - min);
	ptr[j] = value > 255.0 ? (unsigned char)255 : (unsigned char)value;
      }
    }
    
    // rescale the output to the output size
    cv::resize( tmp, dst, output_size);

    // outputTensor should de-allocate here automatically
    
    return(0);
  }

 private:
  // height and width of the most recent input
  int height_ = 0;
  int width_ = 0;

  // height and width of the most recent output
  int out_height_ = 0;
  int out_width_ = 0;

  // netork path and input/output layer names
  char network_path_[256];
  char input_names_[256]; // use Netron.app to see the name of the first layer
  char output_names_[256]; // use Netron.app to see the name of the last layer

  // ORT variables
  Ort::Env env;
  Ort::Session *session_;

  // input data and input tensor variables
  float *input_data = NULL;
  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 3, height_, width_ }; // batch, channel, height, width: 3-channel color image
  
};
