/**
* Monocular-only version of vslamlab_orbslam3_mono_vi.cc
* No IMU support. For use in VSLAM-LAB benchmarking.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>
#include <dirent.h>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

namespace ORB_SLAM3{
    using Seconds = double;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

std::string paddingZeros(const std::string& number, const size_t numberOfZeros = 5){
    std::string zeros{};
    for(size_t iZero{}; iZero < numberOfZeros - number.size(); ++iZero)
        zeros += "0";
    return (zeros + number);
}

void removeSubstring(std::string& str, const std::string& substring) {
    size_t pos;
    while ((pos = str.find(substring)) != std::string::npos) {
        str.erase(pos, substring.length());
    }
}

double ttrack_tot = 0;
int main(int argc, char *argv[])
{
    
    
    const int num_seq = 1;
    
    // ORB_SLAM3  inputs
    string sequence_path;
    string calibration_yaml;
    string rgb_txt;
    string exp_folder;
    string exp_id{"0"};
    string settings_yaml{"orbslam2_settings.yaml"};
    bool verbose{true};
    
    string vocabulary{"Vocabulary/ORBvoc.txt"};

    // Argument parsing: must start with 'run' and use new argument names
    if (argc < 5 || std::string(argv[1]) != "run") {
        std::cerr << "Usage: " << argv[0] << " run vocabulary: <vocab> config: <config.yaml> calibration: <calibration.yaml> sequence: <sequence_path> [other args...]" << std::endl;
        return 1;
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("vocabulary: ") == 0) {
            vocabulary = arg.substr(12);
            std::cout << "[mono] Path to vocabulary = " << vocabulary << std::endl;
            continue;
        }
        if (arg.find("config: ") == 0) {
            settings_yaml = arg.substr(8);
            std::cout << "[mono] Path to config = " << settings_yaml << std::endl;
            continue;
        }
        if (arg.find("calibration: ") == 0) {
            calibration_yaml = arg.substr(13);
            std::cout << "[mono] Path to calibration.yaml = " << calibration_yaml << std::endl;
            continue;
        }
        if (arg.find("sequence: ") == 0) {
            sequence_path = arg.substr(10);
            std::cout << "[mono] Path to sequence = " << sequence_path << std::endl;
            continue;
        }
        if (arg.find("gui: ") == 0) {
            verbose = bool(std::stoi(arg.substr(5)));
            std::cout << "[mono] Activate Visualization = " << verbose << std::endl;
            continue;
        }
    }

    // Check required arguments
    if (vocabulary.empty() || settings_yaml.empty() || calibration_yaml.empty() || sequence_path.empty()) {
        std::cerr << "Error: Missing required argument. Usage: " << argv[0] << " run vocabulary:<vocab> config:<config.yaml> calibration:<calibration.yaml> sequence:<sequence_path>" << std::endl;
        return 1;
    }
    
    // List images in sequence_path (assume images are in a directory, sorted)
    std::vector<std::string> imageFilenames;
    {
        std::vector<std::string> files;
        DIR *dir;
        struct dirent *ent;
        std::cout << "Reading images from: " << sequence_path << std::endl;
        if ((dir = opendir(sequence_path.c_str())) != NULL) {
            std::cout << "Opened directory successfully." << std::endl;
            while ((ent = readdir(dir)) != NULL) {
                std::string fname = ent->d_name;
                if (fname == "." || fname == "..") continue;
                // Accept common image extensions
                if (fname.find(".png") != std::string::npos || fname.find(".jpg") != std::string::npos || fname.find(".jpeg") != std::string::npos){
                    files.push_back(fname);
                    std::cout << fname << std::endl;
                }
            }
            closedir(dir);
        } else {
            std::cerr << "Could not open sequence directory: " << sequence_path << std::endl;
            return 1;
        }
        std::sort(files.begin(), files.end());
        for (const auto& f : files) imageFilenames.push_back(sequence_path + "/" + f);
    }

    int nImages = imageFilenames.size();
    if (nImages <= 0) {
        std::cerr << "ERROR: No images found in sequence directory." << std::endl;
        return 1;
    }

    
    
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabulary,calibration_yaml,settings_yaml,ORB_SLAM3::System::MONOCULAR, verbose);
    float imageScale = SLAM.GetImageScale();
    
    // Read fps from config file (YAML)
    float fps = SLAM.GetFPS();
    
    // Generate timestamps
    std::vector<ORB_SLAM3::Seconds> timestamps;
    for (int i = 0; i < nImages; ++i) {
        timestamps.push_back(i / fps);
    }
    
    double t_resize = 0.f;
    double t_track = 0.f;
    
    int proccIm=0;

    cout.precision(17);
    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        im = cv::imread(imageFilenames[ni],cv::IMREAD_UNCHANGED);
        double tframe = timestamps[ni];
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << imageFilenames[ni] << endl;
            return 1;
        }
        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.TrackMonocular(im,tframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        ttrack_tot += ttrack;
        vTimesTrack[ni]=ttrack;
        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = timestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-timestamps[ni-1];
        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }
    // Stop all threads
    SLAM.Shutdown();
    // Save camera trajectory
    // string resultsPath_expId = exp_folder + "/" + paddingZeros(exp_id);
    // SLAM.SaveTrajectoryEuRoC(resultsPath_expId + "_" + "CameraTrajectory.txt");
    // SLAM.SaveKeyFrameTrajectoryTUM(resultsPath_expId + "_" + "KeyFrameTrajectory.txt");    
    return 0;
}
