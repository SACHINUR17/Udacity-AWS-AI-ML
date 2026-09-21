#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/calculates_results_stats.py
#                                                                             
# PROGRAMMER: RASMO WANYAMA
# DATE CREATED: 07/07/2023                                
# REVISED DATE: 12/07/2023
# PURPOSE: Create a function calculates_results_stats that calculates the 
#          statistics of the results of the program run using the classifier's model 
#          architecture to classify the images. This function will use the 
#          results in the results dictionary to calculate these statistics. 
#          This function will then put the results statistics in a dictionary
#          (results_stats_dic) that's created and returned by this function.
#          This will allow the user of the program to determine the 'best' 
#          model for classifying the images. The statistics that are calculated
#          will be counts and percentages. Please see "Intro to Python - Project
#          classifying Images - xx Calculating Results" for details on how to
#          calculate the counts and percentages for this function.
#         This function inputs:
#            -The results dictionary as results_dic within calculates_results_stats 
#             function and results for the function call within main.
#         This function creates and returns the Results Statistics Dictionary -
#          results_stats_dic. This dictionary contains the results statistics 
#          (either a percentage or a count) where the key is the statistic's 
#           name (starting with 'pct' for percentage or 'n' for count) and value 
#          is the statistic's value. See comments above and the previous topic 
#          Calculating Results in the class for details on how to calculate 
#          the counts and statistics.
##

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using the classifier's model 
    architecture to classifying pet images. Then puts the results statistics in a 
    dictionary (results_stats_dic) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifier labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats_dic - Dictionary that contains the results statistics (either
                    a percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value. See comments above
                     and the previous topic Calculating Results in the class for details
                     on how to calculate the counts and statistics.
    """   
    results_stats_dic = dict()
    
    # Calculate number of images
    n_images = len(results_dic)
    
    # Calculate number of correctly classified dog images
    n_correct_dogs = 0
    for key, values in results_dic.items():
        if values[3] == 1 and values[4] == 1:
            n_correct_dogs += 1
    
    # Calculate number of correctly classified dog breeds
    n_correct_breed = 0
    for key, values in results_dic.items():
        if values[3] == 1 and values[2] == 1:
            n_correct_breed += 1
    
    # Calculate number of dog images
    n_dogs_img = 0
    for key, labels in results_dic.items():
        if labels[3] == 1:
            n_dogs_img += 1
    
    # Calculate number of "not-a-dog" images
    n_notdogs_img = n_images - n_dogs_img
    
    # Calculate percentage of correctly classified dogs
    pct_correct_dogs = (n_correct_dogs / n_dogs_img) * 100 if n_dogs_img > 0 else 0.0
    
    # Calculate percentage of correctly classified dog breeds
    pct_correct_breed = (n_correct_breed / n_dogs_img) * 100 if n_dogs_img > 0 else 0.0
    
    # Calculate percentage of correctly classified "not-a-dog" images
    pct_correct_notdogs = ((n_images - n_dogs_img) / n_notdogs_img) * 100 if n_notdogs_img > 0 else 0.0
    
    # Populate the results_stats_dic dictionary
    results_stats_dic['n_images'] = n_images
    results_stats_dic['n_dogs_img'] = n_dogs_img
    results_stats_dic['n_notdogs_img'] = n_notdogs_img
    results_stats_dic['n_correct_dogs'] = n_correct_dogs
    results_stats_dic['n_correct_breed'] = n_correct_breed
    results_stats_dic['pct_correct_dogs'] = pct_correct_dogs
    results_stats_dic['pct_correct_breed'] = pct_correct_breed
    results_stats_dic['pct_correct_notdogs'] = pct_correct_notdogs
    
    return results_stats_dic
