@echo off
set input_folder=Z://Collaboration_data/Iordonova_lab/downsampled_align/downsampled.tiff
set output_folder=M://Brain_Registration/brainreg_output/notebook_output/

brainreg %input_folder% %output_folder% -v 80 36 36 --orientation sal --atlas whs_sd_rat_39um --save-original-orientation
if errorlevel 1 (
    echo Error: %errorlevel%
    pause
)
