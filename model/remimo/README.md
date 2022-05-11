# RE-MIMO

To replicate the experiment of RE-MIMO on fully correlated channels for QAM-16 modulation, follow the mentioned steps.
1. We need only one RE-MIMO network for all the values of number of users starting from NT=16 to NT=32 for NR=64.
2. Start the training of RE-MIMO by running the command, Python train_classifier.py
3. Track the intermediate validation results, by running the command cat validtn_results/curr_accr.txt or by streaming the print std output to a desired log file
4. During training, the intermediate trained model is stored after each training epoch, in the directory validtn_results by the name model.pth
4. After the RE-MIMO is trained, to test the RE-MIMO, run the command Python test_network.py
7. The test results are stored in the final_results directory. 
8. Before plotting results make sure that the test results of all the other detection methods which is supposed to be in the plot is present/copied to the validtn_results of the RE-MIMO direcory for joint plotting the results in the same plot altogether.
9. If test results of all the required methods are not present in the validtn_results directory of RE-MIMO, please copy them before proceeding to plotting
10. Plot the test results by running the command, Python plot_test_results.py
11. Test result plots are stored in the final_results directory