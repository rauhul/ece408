# Objective

The purpose of this lab is to get you familiar with using the submission system for this course and the hardware used.

# Instructions

Click on the Code tab and then read the code written. Do not worry if you do not understand all the details of the code (the purpose is to get you familiar with the submission system). Once done reading, click the “Compile & Run” button.

The submission system will automatically switch to the compile-and-run results that will also be available through the Attempts tab. There, you will be able to see a summary of your attempt.

The `Timer Output` section has 4 columns:
- Kind describes the type of device whose execution time is being measured. This is specified with the first argument supplied to wbTimer_start. In this example, the GPU execution time is measured.
- Location indicates where (file::line_number) in the user code was the wbTimer_start called. In this example, you should be able to find the line at which wbTimer_start is called in the ‘Program Code’ section with the line_number.
- Time (ms) shows the amount of time it took for the device to run your assignment program, measured in milliseconds.
- Message gives the strong that you passed to wbTimer_start in the second argument. This is typically a comment about the nature of the measurement.

Similarly, you will see the following information under the Logger Output section.

The `Logger Output` section has 3 columns:

- Level is the level specified when calling the wbLog function (indicating the severity of the event),
- Location describes the function::line_number of the wbLog call, and
- Message which is the message specified for the wbLog function

The `Timer Output` or `Logger Output` seconds are hidden, if no timing or logging statements occur in your program.

We log the hardware information used for this course --- the details which will be explained in the first few lectures.
- GPU card’s name
- GPU computation capabilities
- Maximum number of block dimensions
- Maximum number of grid dimensions
- Maximum size of GPU memory
- Amount of constant and share memory
- Warp size

All results from previous attempts can be found in the Attempts tab. You can choose any of these attempts for submission for grading. Note that even though you can submit multiple times, only your last submission will be reflected in the course database.

After completing this lab, and before proceeding to the next one, you will find it helpful to watch the tutorial videos if you have not done so.

Click the ‘Questions’ tab and answer the questions to the best of your ability. Some of the questions will be better answered after the first few lectures. You could use Google to find most of the answers. No points are assigned to questions in this assignment.
