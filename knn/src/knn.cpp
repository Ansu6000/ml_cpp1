#include "/workspaces/ml_cpp1/knn/include/knn.h"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "/workspaces/ml_cpp1/include/data_handler.h"
#include "/workspaces/ml_cpp1/include/data.h"
#include "/workspaces/ml_cpp1/include/common.h"


knn::knn(int val)
{
    k = val;
}
knn::knn()
{
    // Nothing
}
knn::~knn()
{
    // Nothing
}

void knn::find_knearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();
    double previous_min = min;
    int index = 0;
    for(int i = 0; i < k; i++)
    {
        if(i == 0)
        {
            for(int j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                if(distance < min)
                {
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        } else
        {
            for(int j = 0; j < training_data->size(); j++)
            {
                double distance = training_data->at(j)->get_distance();
                if(distance > previous_min && distance < min)
                {
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}


void knn::set_k(int val)
{
    k = val;
}

int knn::predict()
{
    std::map<uint8_t, int> class_freq;
    for(int i = 0; i < neighbors->size(); i++)
    {
        if(class_freq.find(neighbors->at(i)->get_label()) == class_freq.end())
        {
            class_freq[neighbors->at(i)->get_label()] = 1;
        } else
        {
            class_freq[neighbors->at(i)->get_label()]++;
        }
    }

    int best = 0;
    int max = 0;
    for(auto kv : class_freq)
    {
        if(kv.second > max)
        {
            max = kv.second;
            best = kv.first;
        }
    }
    delete neighbors;
    return best;
}

double knn::calculate_distance(data *query_point, data *input)
{
    double distance = 0.0;
    if(query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        printf("Error Vector Size Mismatch. \n");
        exit(1);
    }
    for(unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
    distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);
    return distance;    
}
double knn::validate_performance()
{
    double current_performance = 0.0;
    int count = 0;
    int data_index = 0;
    for(data *query_point : *validation_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if(prediction == query_point->get_label())
        {
            count++;
        }
        data_index++;
        printf("Current Performance = %.3f %%\n", ((double)count*100.0)/((double)data_index));
    }
    current_performance = ((double)count*100.0)/((double)validation_data->size());
    printf("Validation Performance for K = %d: %.3f %%\n", k, current_performance);
    return current_performance;
}
double knn::test_performance()
{
    double current_performance = 0.0;
    int count = 0;
    for(data *query_point : *test_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if(prediction == query_point->get_label())
        {
            count++;
        }
    }
    current_performance = ((double)count*100.0/((double)test_data->size()));
    printf("Tested Performance = %.3f %%\n", current_performance);
    return current_performance;
}
