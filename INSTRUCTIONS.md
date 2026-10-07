1. Project Overview

In this project, you will develop a compact machine-learning/deep-learning system for ECG heartbeat classification.

The project represents the first, ECG-only stage of a larger wearable cardiac-monitoring problem. No motion sensor or IMU data will be used.

The intended deployment platform is the Analog Devices MAX78002 Edge-AI microcontroller. However:

    No physical MAX78002 development kit will be provided or required.

Your task is therefore to develop a model that is compatible with the MAX78002 deployment workflow, demonstrate software-level quantization and synthesis, and critically evaluate its performance.

The project must be carried out according to the CRISP-DM methodology:

    Business/Application Understanding

    Data Understanding

    Data Preparation

    Modeling

    Evaluation

    Deployment

CRISP-DM is iterative. You are expected to return to earlier stages when later experiments reveal problems.
2. Problem

The goal is to classify ECG heartbeats into clinically meaningful heartbeat categories while satisfying the computational constraints of an Edge-AI device (maximum model size 2MB).
Required classification task

Use the MIT-BIH Arrhythmia Database and perform four-class heartbeat classification:

    N — Normal-type beats

    S — Supraventricular ectopic beats

    V — Ventricular ectopic beats

    F — Fusion beats

Use the standard MIT-BIH annotations to construct the classes.

Rare paced/unclassifiable beats that do not belong to the four required classes may be excluded, but all exclusions must be documented.
Important

You are not developing a medical device and you must not claim that your model is clinically validated.

The purpose is to study the complete data-mining and Edge-AI development process.
3. Main Dataset
MIT-BIH Arrhythmia Database

The primary dataset contains:

    48 approximately 30-minute ECG recordings

    47 subjects

    two ECG channels

    sampling frequency of 360 Hz

    expert beat annotations

    approximately 110,000 beat annotations

For the main experiment, use a patient/record-independent training and testing strategy.

Do not randomly distribute beats from the same recording between training and testing.
Recommended test protocol

Use the commonly used inter-patient MIT-BIH partition.
DS1 — Development set

101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230

Use DS1 for:

    training

    validation

    model selection

    hyperparameter selection

The internal training/validation split must again be performed by record, not randomly by heartbeat.
DS2 — Final test set

100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

DS2 must remain untouched until final evaluation.
4. Robustness Dataset
MIT-BIH Noise Stress Test Database

Real ECG recordings contain noise caused by factors such as:

    baseline wander

    muscle/EMG activity

    electrode movement

The Noise Stress Test Database provides representative ECG noise signals.

Your final system must therefore include a noise-robustness experiment.

At minimum, evaluate the model under:

    clean ECG

    moderate noise

    strong noise

Recommended SNR conditions include approximately:

    12 dB

    6 dB

    0 dB

You may add noise from the Noise Stress Test Database to held-out ECG signals.

The noise experiment must never be used to leak information from the final test set into model training.
5. Edge-AI Constraint

The target platform is the:

Analog Devices MAX78002

The model should therefore be designed as an embedded/Edge-AI model rather than an unrestricted datacenter model.

Prefer operations supported by the MAX78002 CNN accelerator, particularly:

    1D convolution

    pooling

    ReLU-type activation

    compact fully connected output layers

    fixed-size inputs

Avoid architectures that cannot reasonably be deployed using the MAX78002 CNN accelerator.

Examples of architectures that should generally be avoided for this assignment include unrestricted:

    Transformers

    very large ResNets

    LSTMs/GRUs

    models requiring hundreds of MB of parameters

    unsupported custom operations

A small 1D CNN is an appropriate starting point.

You may develop a better architecture if you can demonstrate that it remains compatible with the MAX78002 toolchain.
6. Required Experiments

At minimum, your final project must contain:

    A simple baseline

    A compact neural-network model

    FP32 model results

    INT8/quantized model results

    Patient-independent evaluation

    Per-class evaluation

    Noise robustness evaluation

    MAX78002 software synthesis/deployment analysis

You are encouraged to investigate issues such as:

    class imbalance

    normalization

    filtering

    ECG window size

    data augmentation

    loss functions

    model size

    quantization-aware training

    robustness versus model complexity

PROJECT SUBMISSION Task 1: CRISP-DM: Problem, Data and Preparation

Submission 1 covers the first three CRISP-DM stages.

Your report must use the following sections.
1. Business/Application Understanding
1.1 Application Scenario

Explain:

    What problem are you trying to solve?

    Where could automatic ECG classification be useful?

    Why might inference directly on an Edge-AI device be useful?

    Who could potentially use such a system?

1.2 Data-Mining Objective

Clearly state the machine-learning problem.

Complete:

Input:
[Describe exactly what the model receives.]

Output:
[Describe the classes predicted by the model.]

Type of task:
[Classification, multiclass classification, etc.]
1.3 Project Objectives

List your technical objectives.

For example:

    distinguish N/S/V/F heartbeats

    generalize to unseen ECG recordings

    remain robust to ECG noise

    develop a compact model

    make the model compatible with MAX78002

1.4 Success Criteria

Define how you will decide whether the project has succeeded.

Include criteria for:

Predictive performance

Minority-class performance

Noise robustness

Model complexity

MAX78002 compatibility

You must define measurable criteria rather than simply stating "high accuracy".
1.5 Constraints

Discuss:

    limited Edge-AI memory

    fixed-point/INT8 computation

    class imbalance

    limited number of subjects

    noise

    computational resources

    absence of physical MAX78002 hardware

1.6 Risks and Challenges

Identify at least three important risks.

For each:
Risk 	Why is it important? 	Proposed mitigation
		
		
		
2. Data Understanding
2.1 Dataset Description

Report:

    database name

    number of records

    number of subjects

    number of channels

    sampling frequency

    recording duration

    annotation type

    ECG leads available

2.2 Target Classes

Create a table containing:
Final class 	Original annotations included 	Number of beats
N 		
S 		
V 		
F 		

Explain any annotations that you excluded.
2.3 Exploratory Data Analysis

At minimum, show:

    example ECG signal

    example heartbeat from each class

    class distribution

    subject/record distribution

    signal amplitude distribution

    examples of noisy ECG

2.4 Data Quality

Investigate:

    missing data

    extreme amplitudes

    corrupted segments

    class imbalance

    differences between recordings

    differences between ECG channels

    possible noisy regions

2.5 Leakage Analysis

Explain how data leakage could occur in this project.

Specifically explain why:

    Randomly splitting individual heartbeats from the same subject into training and test sets can produce misleading results.

State exactly how your approach prevents this.
3. Data Preparation
3.1 Record Selection

List exactly which records are used for:

    training

    validation

    testing

3.2 ECG Channel Selection

State:

    which ECG channel/lead you use

    why you selected it

    what you do if the same lead is not available in a recording

3.3 Beat Extraction

Describe:

    how R-peak annotations are obtained

    samples before the annotated beat

    samples after the annotated beat

    final input length

Your network must receive a fixed-size input.
3.4 Signal Preprocessing

State exactly which operations are performed and in which order.

For example:

Raw ECG
→ filtering
→ segmentation
→ normalization
→ clipping
→ model input

For every preprocessing operation explain why it is necessary.
3.5 Label Preparation

Document the complete mapping:

Original MIT-BIH annotation
→ final N/S/V/F class.
3.6 Class Imbalance

Report the imbalance.

Explain whether you will use:

    class weights

    oversampling

    undersampling

    balanced batches

    focal loss

    another strategy

    no balancing

Justify your decision.
3.7 Data Augmentation

Describe any augmentation used during training.

Possible examples include:

    amplitude scaling

    small temporal shifts

    ECG noise

    baseline wander

Augmentation must only be applied to the training data.
3.8 Final Dataset

Report:
Split 	Subjects/records 	N 	S 	V 	F 	Total
Train 						
Validation 						
Test 						
3.9 Reproducibility

State:

    software versions

    random seed

    preprocessing parameters

    sampling frequency

    input size

Submission 1 Deliverables

Submit:

    CRISP-DM Report — Stages 1–3

    EDA notebook

    Data-preparation code

    Generated train/validation/test statistics

    README describing how to reproduce the preprocessing

No final neural-network results are required at this stage.
PROJECT SUBMISSION Task 2: CRISP-DM: Modeling, Evaluation and Deployment

Submission 2 completes the project.

You may revise your conclusions from Submission 1 if your experiments reveal problems. Document significant changes and explain why they were made.
4. Modeling
4.1 Baseline

Implement at least one simple baseline.

Examples:

    Logistic Regression

    Random Forest

    SVM
    CNN

    very small neural network

Report its performance.
4.2 Proposed Edge Model

Describe your final neural-network architecture.

Provide a table:
Layer 	Input 	Operation 	Output 	Parameters
				

Explain why this architecture is appropriate for Edge-AI.
4.3 Training Configuration

Report:

    optimizer

    learning rate

    batch size

    epochs

    loss function

    early stopping

    class weighting

    augmentation

    random seed

4.4 Model Selection

Explain:

    which models you tried

    which hyperparameters were varied

    which validation metric selected the final model

    why the final architecture was selected

4.5 Quantization

Compare:

FP32 model

versus

INT8/quantized model

If Quantization-Aware Training is used, explain:

    why QAT is needed

    how it was performed

    whether quantization changed model performance

5. Evaluation

Do not report accuracy alone.

At minimum report:

    Accuracy

    Macro F1-score

    Per-class Precision

    Per-class Recall/Sensitivity

    Per-class F1-score

    Confusion matrix

5.1 Clean Test Results
Metric 	Baseline 	FP32 Edge Model 	Quantized Model
Accuracy 			
Macro F1 			
Model parameters 			
5.2 Per-Class Performance
Class 	Precision 	Sensitivity/Recall 	F1
N 			
S 			
V 			
F 			

Discuss which classes are difficult and why.
5.3 Quantization Analysis

Report:

FP32 Macro F1: ______

INT8 Macro F1: ______

Difference: ______

Discuss the effect of quantization.
5.4 Noise Robustness

Evaluate the final model under progressively increasing ECG noise.
Condition 	Accuracy 	Macro F1
Clean 		
12 dB 		
6 dB 		
0 dB 		

Plot performance versus noise level.

Discuss:

    which classes are most affected

    whether degradation is gradual

    whether preprocessing helps

    whether robustness could be improved

5.5 Error Analysis

Select representative errors and investigate them.

Include examples such as:

    N classified as S

    S classified as N

    V classified incorrectly

    errors under strong noise

Do not only show numerical results. Examine the ECG waveforms.
5.6 Success-Criteria Review

Return to the success criteria defined in Submission 1.

For every criterion report:

Achieved / Partially achieved / Not achieved

and justify your conclusion.
6. Deployment

For this course, "deployment" means software-level preparation for deployment to the MAX78002.

A physical development kit is not required.
6.1 MAX78002 Compatibility

Explain:

    why the architecture can run on the MAX78002 CNN accelerator

    which layer types are used

    input dimensions

    output dimensions

    model parameter count

6.2 MAX78002 Toolchain

Use the Analog Devices:

    ai8x-training framework

    ai8x-synthesis framework

Attempt to:

    prepare the model using the MAX78002-compatible training framework

    quantize the model

    synthesize the model for MAX78002

    generate the device-specific output/C code

6.3 Deployment Evidence

Provide evidence of the synthesis process.

Report:
Requirement 	Result
Model accepted by MAX78002 toolchain 	PASS / FAIL
Quantization completed 	PASS / FAIL
Synthesis completed 	PASS / FAIL
C/device files generated 	PASS / FAIL
Weight-memory requirement 	
Data-memory requirement 	

If synthesis fails, identify exactly why and modify the architecture if possible.

A model that obtains excellent accuracy but cannot be synthesized for the MAX78002 does not fully satisfy the project objective.
6.4 Hardware Limitations

Because no physical board is supplied, you must not claim measured hardware values for:

    inference latency

    power consumption

    energy per inference

    real MAX78002 accuracy

    real-time performance

Clearly distinguish:

Software verified

from

Hardware verified.
6.5 Proposed Hardware Validation

Briefly explain how you would test the model if a MAX78002 kit became available.

Include:

    loading the generated model

    supplying identical ECG test inputs

    comparing Python and hardware outputs

    checking intermediate/final output consistency

    measuring inference time

    measuring energy consumption

7. Final Discussion

Discuss:

    What did you learn from the data?

    What was the largest source of error?

    Was the highest-accuracy model also the best Edge-AI model?

    How much performance was lost after quantization?

    How sensitive was the system to ECG noise?

    What compromises were necessary because of hardware constraints?

    What would you change with more time?

8. Limitations

At minimum discuss:

    limited dataset size

    class imbalance

    differences between patients

    use of historical ECG datasets

    absence of prospective clinical testing

    absence of hardware testing

    possible domain shift to wearable ECG devices

9. Final Submission Package

If not using python, submit equivalent structure with your files.

Submit one compressed project folder containing:

GroupXX_ECG_Project/
│
├── report.pdf
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── prepare_data.py 
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── noise_test.py
│
├── configs/
│
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix.*
│   └── noise_results.*
│
└── max78002/
    ├── model_configuration
    ├── quantization_output
    └── synthesis_output

Do not submit the complete raw PhysioNet datasets.

Your README must provide instructions for downloading the required data.
10. Recommended Project Resources

Students should use the following resources supplied on the course page:
Required

    MIT-BIH Arrhythmia Database — PhysioNet

    MIT-BIH Noise Stress Test Database — PhysioNet

    WFDB Python package

    MAX78002 product documentation

    Analog Devices MAX78000/MAX78002 AI documentation

    Analog Devices ai8x-training repository

    Analog Devices ai8x-synthesis repository

    CRISP-DM reference material

Optional

    European ST-T Database

    additional published literature on ECG classification and Edge-AI

Important Rules

    Do not randomly mix beats from the same ECG recording across training and testing.

    Do not use the final DS2 test set for model selection.

    Do not optimize hyperparameters using test-set performance.

    Do not report accuracy alone.

    Do not claim clinical validation.

    Do not claim measurements from MAX78002 hardware when no board was used.

    Your submitted model must be designed for MAX78002 compatibility.

    All preprocessing, exclusions and class mappings must be reproducible.

    The project must follow and explicitly document all six CRISP-DM stages.

From canvas:

Clarification on Project Scope and Edge-AI Requirements

Dear Students,

Some questions have come up regarding the Edge-AI requirements and the expected scope of the project, so I would like to clarify them for everyone.

The main purpose of the project is to apply the data-mining workflow and understand how model-development decisions are affected by real-world Edge-AI constraints.

A few points to keep in mind:

    Edge-AI constraints: You should consider factors such as model size, number of parameters, computational requirements, memory usage, and model precision. However, you are not expected to produce a fully deployed industrial solution.

    2 MB model-size constraint: The 2 MB limit comes from the real use case that inspired this project. For the course, it should be treated as a realistic design target rather than a strict pass/fail requirement. You are not required to reduce the model size to 2mb, but do quantization and compare. at least try 16 and 8 bit Int. 

    Quantization: You are expected to perform quantization, preferably INT8 where appropriate, and compare the model before and after quantization. The comparison should include aspects such as model size, predictive performance, and the trade-off between efficiency and accuracy.

    RapidMiner: You may use RapidMiner for suitable parts of the project, such as data exploration, preprocessing, modelling, and evaluation. If a particular requirement, such as quantization, is not conveniently supported within your RapidMiner workflow, you may use another suitable framework or tool for that part.

    MAX78002 compatibility, software synthesis, and deployment: You are not required to physically deploy your model on a MAX78002 device. You should, however, understand the relevant hardware constraints and documentation and discuss whether your proposed model would be suitable for such an Edge-AI deployment.

The main emphasis should remain on data understanding, preprocessing, modelling, validation, evaluation, and critical analysis. The hardware-related requirements are included to expose you to realistic deployment considerations and encourage you to think beyond predictive accuracy alone.