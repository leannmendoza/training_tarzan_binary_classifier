# Training Tarzan: A Machine Learning Simulation (Based on Netflix’s Start-Up)

**Author:** LeAnn Mendoza
**Course:** DS5010: Intro to Programming for Data Science
**Semester:** Spring 2022
**Campus:**  Northeastern University, Silicon Valley

![enter image description here](https://i.imgur.com/HBr50Ph.png)
# Package Description

This program simulates Do-San’s Tarzan analogy of machine learning  [(link to clip)](https://youtube.com/clip/Ugkxs_VCFjTgGugzAIkanCq1qtxetKY99Vog)  using a Tensorflow binary image classifier. The purpose of this project is to simulate the concept of machine learning using deep learning and image classification.

## Features

 1.   [Driver program](https://github.com/leannmendoza/training_tarzan_binary_classifier/blob/main/driver.py) walks the user interactively through the ML process from data preprocessing, to training, and analyzing results.
		 - [Video demo of driver.py](https://youtu.be/7zyWgmHuD0w)
 2.   Visualizations of class distributions (bar graphs)

 3.   Quickly pull up on image and its label (truth and/or predictions) at any point in the process
 4.   Analysis tools and visualizations such as confusion matricies and auroc curves.
 ![Confusion Matrix result
](https://lh3.googleusercontent.com/fJmIxSzanwPASp12FgyJ_rROmD82qCrVrLPQoXl2nhtgF6AHHyxV9N1t8j0a7MwjHzR30qjzkhf4MLWVTBcJ_Ye750gVhYmK3GZuyEnuYRZpC2K5o-LFcxym2Jg4EowAWTug5AhkBkLgusp-WsqHQOsOttANMD8hUsIzp9FGcypMH6bwqZLSbJgFm_wZQynVcbCWgFHQcOUNDIfGFJk5gS5FuI_ASdh74blVIMy2iRcG9lpTkNZWRbS2qmLJB9VjL9o_RPseKKEIGjwEhR8Kxe1eZu89GP8CCdYxDZer6htK5eat7q86BZox36ATtKsrDJc6TD4Dyl-xwg4Pe9aP7Fmit6aLyVPoaXREBLmDv-iJ67ioUZTvO1I3oSyEogl1snRQGrHsckxPi3YZ4xyHhaxeaWaTiVqQsBaVbZ0vD9wHn-W4KOhkJ7HhyXgAQ31HMAFDnyEtcXSFQ_NPnLnUTFn409TDT1A3rEd83IReBv9IO3afUnrPk595hyMOX-NvY8qS_nGRel64KWpWYRwfS7sMF_hkmTwHBxI59auO0G1SuGUTlLSdu_CUw2Ewm9RcgAFbd62u8_MQ9tGSx47Kf41bvlHF8cVg1QzWZpOhY63IZGvin2gSb2CKKMjgBv1zhQ5jClLfLYNl2meU8B0yCjfI_SikHzgGAK8RV9_8EG65yuL9xLuDx07apQtQfZdE9TqLpLsVJlgM8p8VlBvz4b6JLinvOznfqmnf5PoL1rHTy-GlJ3irbE_4ZUGtEngvntS5CBr2zvaTPwcSbwXK-o0DjEw8rmoh9WQIHYKcwS1mpR5U04BOcwASfb-FC4QMi4tJq-k=w640-h480-no?authuser=1)
![auroc curve](https://lh3.googleusercontent.com/4SoeaYbq8SkFs-zqwXB3aZWRFt9bBXil94i2Mgt6QsaRGdRfgXL2N87aERpA0RSq8moQS6jjSB5en0fEc-d2HS_W9-O9PMJQBDFsQRq8qluLD7xRddQfVOH4gP7h-RC2rKqLNzXGUCg8ql31Mn5oZ10cGac_1o9ut46Flu0HwrQ3UMABCX7O74bgnwZuky_xsZpapfl1Z441is4Q9ifawaqlpxyK0PInuW-K1Phc_ojk3-KvkNlrzKeQsRTsBGny-X2Q_qzlDjax6xSbXhtovwP6JlgTi40HC1SaD4iLLmv5iNDufWr63rTvpHcPntgtSS6nuundg9VEyM7Je6Uu2990sjf2Llguw-1Cb0dz2h9f1Bcrlpdj8uerPwUgrbkLk25Ot1VKXbcRSqi643HRrIdJrcXwHMwYXAbx2Z9W423HfuM6TilwfUAgH3KLmg4nREA0caXasd8p1Q9uZCD6NKYpIGiwgVYiHhffc8cZLJexNBsiiyt9pE3Jyx8yoE1OZxIPQAGZQvOkM0Ua8Tl67M14AlgqOU694izc8ndgRBjDvzRCWTfwX-lS_F-fxa_nXjJA6UHOObTxo5Z1zZMtkRyA5T3MjsTtzjZd22pTfyhV2-TYn-VAlUOFO6KFbqA51St06kNHT6UGzsdY5y33GUm1OhKsvUnOoAHPWnLC8sj1pD-AbygKwzzHs0iGZn1j8d8DhIASfGG40tKd0S0ZIQ58Fol2iB5jyiee0WJ3ueav7bpiIsX4Hg1TlsY3Vsu82EI2NrJ2bLROpTyLjkaXrgDWf2x3z8NG7ryA99wReffaEU92BeDJsWgDJRGrflwhIUFpMuk=w640-h480-no?authuser=1)
## Set-up

 1. Clone this repository
 2. Download image data from [google drive](https://drive.google.com/file/d/1l5wUkxZcUa2Fd6SGhqhs_sdZZesleCNk/view), and place in directory
 3. Install requirements.txt using conda (developed and tested on Apple M1)
 4. Activate conda enviroment
 5. Run [driver.py](https://github.com/leannmendoza/training_tarzan_binary_classifier/blob/main/driver.py) 

## Video Tutorial

https://youtu.be/7zyWgmHuD0w

## Code Organization
![enter image description here](https://lh3.googleusercontent.com/nKA23vg2-tJNs3tUdQTYpgCwaFSgjItVlu3c7O7EUyM2Jxesc7dipbfDXHdn068pL6HfFeWLDBZBBpHn_GpuXjMg8_HKxgunXTKZs6MqQ31n1CrRrQa_pNdJXbt3Z1y3-G4YfSmgGoWMr5aq2CP8XcDzXFZfBvwCr5VzIkI1emHeCFrdD1dhRH4PZuBuIfUPC71bu0UeKQo7uceHgQTkaTPFyS97VQF96fjJznToY8TzK5ZPHAH2dWmFB2y9z0hWr2MCaLykTLQrAHW8Vm_hAsGm-f8-sD4-JDLE8z9h2D4_Y09suM0WS87aPUAG1G8xRItYge1FYk2tXMEW_AJXb3DM-vMii7aWVQbcTE_Js_wFook_joQylQoVvWdNT2wcxus9wP7MhSeqSw8ex07tKcD2gJo1gbx0tZbT7sCzKMM2S30A0D7VFYFt24YHrPa4E0sDDxSgtpUU0nqlDWyzNRhzXZGTBkc-kTxWo76bM2DPhBNhzJv7nMsftKx6QI2CZcW0FAk-uFl9fHyIud9rgg8_-R3BL6s3eApTNjsDfN1gCRYEh7fNjwF_cF3v5neQoxbTi_oOju4W_J5xMh5fiYGbvjhuWlAhJ1jPro5veRp8emkEmCfVDExTz8VZ_qEcT45AjHfDWZoZfmNpGDC0Txy96OvrXJf926UjDfKQxHcsK4HkfldlJyX3w0qY_xaNXxpVXtkxdBDlBG4OJxp6HOBxmZ-6JBJNgcZcSC72T2UxVZW0hjouW7X8PHxS2pWXezR81dVUBVsM6VwsU_o1ffJS-JWw9fz2_qkVXclCjQWZvaMDXzeKVFanUl5Y4V-oEPzuejY=w730-h662-no?authuser=1)
