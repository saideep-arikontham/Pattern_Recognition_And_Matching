// -------------------------------------------------------------------------------------------
// Saideep Arikontham
// Feb 2025
// Pattern Recognition and Computer Vision - Project 2
// -------------------------------------------------------------------------------------------


#include <cstdlib>
#include <dirent.h>
#include <cstdio>   // printf
#include <cstring>  // strcpy, strcmp
#include <opencv2/opencv.hpp>  // OpenCV
#include "opencv2/dnn.hpp"  
#include "csv_util.h"  // CSV utility functions
#include "utils.h"  // Utility functions for feature vectors


int main(int argc, char *argv[]) {
    cv::Mat src;
    char target_img_file[256];
    char feature_set[256];
    char feature_vector_file[256];

    // ------------------------------------------------------------------------
    // Command-line argument validation
    // ------------------------------------------------------------------------
    if (argc < 4) {
        printf("Usage: %s <target image> <feature set> <feature vector file>\n", argv[0]);
        exit(-1);
    }

    // Safe string copy
    strncpy(target_img_file, argv[1], 255);
    strncpy(feature_set, argv[2], 255);
    strncpy(feature_vector_file, argv[3], 255);

    // Read and display the target image
    read_image(target_img_file, src);
    display_image(target_img_file, src);

    // Feature vectors and filenames storage
    std::vector<float> target_img_data;
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;
    std::vector<float> ssd_data;

    // ------------------------------------------------------------------------
    // Feature Extraction & Matching
    // ------------------------------------------------------------------------

    if (strcmp(feature_set, "7x7_middle_square") == 0) {  
        /*
        make generate_feature_vectors
        ./bin/generate_feature_vectors ./olympus 7x7_middle_square
        make pattern_recognition
        ./bin/pattern_recognition ./olympus/pic.1016.jpg 7x7_middle_square ./feature_vectors/feature_vectors_7x7.csv
        */
        get_feature_vector_7x7(target_img_file, target_img_data);
        read_image_data_csv(feature_vector_file, filenames, data);

        for (int i = 0; i < data.size(); i++) {
            float ssd = sum_of_squared_difference(target_img_data, data[i]);
            ssd_data.push_back(ssd);
        }

        get_top_N_matches(filenames, ssd_data, 3, true);
    } 

    else if (strcmp(feature_set, "rg_histogram") == 0) {  
        /*
        make generate_feature_vectors
        ./bin/generate_feature_vectors ./olympus rg_histogram
        make pattern_recognition
        ./bin/pattern_recognition ./olympus/pic.0164.jpg rg_histogram ./feature_vectors/feature_vectors_rg_hist.csv
        */
        get_feature_vector_rg_hist(target_img_file, target_img_data);
        read_image_data_csv(feature_vector_file, filenames, data);

        for (int i = 0; i < data.size(); i++) {
            float ssd = histogram_intersection(target_img_data, data[i]);
            ssd_data.push_back(ssd);
        }

        get_top_N_matches(filenames, ssd_data, 3, false);
    } 

    else if (strcmp(feature_set, "two_rgb_histogram") == 0) {  
        /*
        make generate_feature_vectors
        ./bin/generate_feature_vectors ./olympus two_rgb_histogram
        make pattern_recognition
        ./bin/pattern_recognition ./olympus/pic.0274.jpg two_rgb_histogram ./feature_vectors/feature_vectors_two_rgb_hist.csv
        */
        get_feature_vector_two_rgb_hist(target_img_file, target_img_data);
        read_image_data_csv(feature_vector_file, filenames, data);

        for (int i = 0; i < data.size(); i++) {
            // Compute intersection for top half (first 512 values)
            std::vector<float> target_top(target_img_data.begin(), target_img_data.begin() + 512);
            std::vector<float> img_top(data[i].begin(), data[i].begin() + 512);
            float intersection_total = histogram_intersection(target_top, img_top);

            // Compute intersection for bottom half (next 512 values)
            std::vector<float> target_bottom(target_img_data.begin() + 512, target_img_data.end());
            std::vector<float> img_bottom(data[i].begin() + 512, data[i].end());
            float intersection_mid = histogram_intersection(target_bottom, img_bottom);

            // Combine both intersections (equally weighted)
            float final_score = intersection_total + intersection_mid / 4;
            ssd_data.push_back(final_score);
        }

        get_top_N_matches(filenames, ssd_data, 3, false);
    } 

    else if (strcmp(feature_set, "rgb_texture_hist") == 0) {  
        /*
        make generate_feature_vectors
        ./bin/generate_feature_vectors ./olympus rgb_texture_hist
        make pattern_recognition
        ./bin/pattern_recognition ./olympus/pic.0535.jpg rgb_texture_hist ./feature_vectors/feature_vectors_rgb_texture_hist.csv
        */
        get_feature_vector_rgb_texture_hist(target_img_file, target_img_data);
        read_image_data_csv(feature_vector_file, filenames, data);

        for (int i = 0; i < data.size(); i++) {
            // Compute intersection rgb histogram (first 512 values)
            std::vector<float> target_rgb(target_img_data.begin(), target_img_data.begin() + 512);
            std::vector<float> img_rgb(data[i].begin(), data[i].begin() + 512);
            float chiSquare_rgb = chiSquareDistance(target_rgb, img_rgb);

            // Compute intersection magnitude histogram (next 512 values)
            std::vector<float> target_texture(target_img_data.begin() + 512, target_img_data.end());
            std::vector<float> img_texture(data[i].begin() + 512, data[i].end());
            float chiSquare_texture = chiSquareDistance(target_texture, img_texture);

            // Combine both intersections (equally weighted)
            float final_score = (chiSquare_rgb + chiSquare_texture) / 2 ;
            ssd_data.push_back(final_score);
        }

        get_top_N_matches(filenames, ssd_data, 3, true);
    } 

    else if (strcmp(feature_set, "deep_network") == 0) {  
        /*
        make pattern_recognition
        ./bin/pattern_recognition ./olympus/pic.0893.jpg deep_network ./feature_vectors/ResNet18_olym.csv
        ./bin/pattern_recognition ./olympus/pic.0164.jpg deep_network ./feature_vectors/ResNet18_olym.csv
        */
        int target_image_index;

        read_image_data_csv(feature_vector_file, filenames, data);

        for (int i=0; i<filenames.size(); i++){
            std::string newPath = std::string("./olympus/") + filenames[i];
            free(filenames[i]);
            filenames[i] = strdup(newPath.c_str());

            if(strcmp(target_img_file, filenames[i]) == 0){
                target_image_index = i;
            }
        }

        target_img_data = data[target_image_index];

        for (int i=0; i<data.size(); i++){
            float ssd = computeCosineDistance(target_img_data, data[i]);
            ssd_data.push_back(ssd);
        }

        get_top_N_matches(filenames, ssd_data, 3, true);
    }

        else if (strcmp(feature_set, "depth_rgb") == 0) {  
        /*
        make generate_feature_vectors
        ./bin/generate_feature_vectors ./images depth_rgb
        make pattern_recognition
        ./bin/pattern_recognition ./images/banana1.jpg depth_rgb ./feature_vectors/feature_vectors_depth_rgb_hist.csv

        make generate_feature_vectors
        ./bin/generate_feature_vectors ./olympus depth_rgb
        make pattern_recognition
        ./bin/pattern_recognition ./olympus/pic.1074.jpg depth_rgb ./feature_vectors/feature_vectors_depth_rgb_hist.csv
        */
        get_feature_vector_depth_rgb_hist(target_img_file, target_img_data);
        read_image_data_csv(feature_vector_file, filenames, data);

        for (int i = 0; i < data.size(); i++) {
            // Compute intersection rgb histogram
            float intersection_score = histogram_intersection(target_img_data, data[i]);

            // append intersection
            ssd_data.push_back(intersection_score);
        }

        get_top_N_matches(filenames, ssd_data, 8, false);
    } 

    else if (strcmp(feature_set, "dnn") == 0) {  
        /*
        make generate_feature_vectors
        ./bin/generate_feature_vectors ./images dnn        
        make pattern_recognition
        ./bin/pattern_recognition ./images/husky1.jpg dnn ./feature_vectors/feature_vectors_dnn.csv
        */

        read_image_data_csv(feature_vector_file, filenames, data);
        get_feature_vector_dnn(target_img_file, target_img_data);

        for (int i=0; i<data.size(); i++){
            float ssd = computeCosineDistance(target_img_data, data[i]);
            ssd_data.push_back(ssd);
        }

        get_top_N_matches(filenames, ssd_data, 5, true);
    }


/*Use deep network vectors along with texture histogram*/

    else {
        printf("Unknown feature set: %s\n", feature_set);
        exit(-1);
    }

    // ------------------------------------------------------------------------
    // Wait for keypress (Press 'q' to quit)
    // ------------------------------------------------------------------------
    loop_until_q_pressed();
    return 0;
}
