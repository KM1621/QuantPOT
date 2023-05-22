# QuantPOT
This repo presents the source code used in "Gradient estimation for ultra low precision POT and additive POT quantization"
## Custom gradient estimation for low bit width quantization
The code for custom gradient estimation and quantization error is inspired by [APoT_Quantization](https://github.com/yhhhli/APoT_Quantization). 
We plan to provide detailed instruction on how to use the code for the custom gradient estimation experiment. In the meantime, if you have any specific questions about the code or need further clarification on our methodology, feel free to share your thoughts or raise any concerns by posting in the 'Issues' tab. We are more than happy to provide additional information to support the understanding and replication of our work. 
## Integer deployable implementation
The code for the integer deployable implementaiton is inspired by NEMO tool, a library for minimization of Deep Neural Networks developed in PyTorch for PULP-based microcontrollers
[NEMO (NEural Minimizer for pytOrch)](https://github.com/pulp-platform/nemo).
Please follow instructions in [NEMO](https://github.com/pulp-platform/nemo), for installation and requirements. In windows, double click nemoCifarbatch.bat file to launch the integer deployable runs, once all requirements are met.
