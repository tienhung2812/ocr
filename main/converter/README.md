# Conver PDF to XML


### **Method 1:** Using Tessearct to convert from Image to XML

#### Step to proceduce:
- Using LSTM from tesseract
- Using ```pytesseract``` or ```Tesseract command``` to convert both using English package and Vietnamese
- Compare **Confidence** on each position -> Replace with language have **higher** confidence

#### Result
- Unreliable because of **Confidence** on english package **usually** have **high** point (>80) but the result is incorrect 
- Most of Vietnamese words is have **lower** confidence and be replaced by English

#### Conclusion
- Not Reliable
- Because of how tesseract is detect words is spilt image using English padding 
    - Example: **cát** 
        - English: Tesseract crop the word without acute accent, but in **cat** in English is have meaning -> High Confidence
        - Vietnamese: esseract crop the word without acute accent or with small sign of its-> **cát** but with Low Confidence

### **Method 2:** Using open cv to split image for preprocessing before send to tesseract

### Reason
- OpenCV have a Text Dector package inside is [**OpenCV’s EAST text detector**](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)

- Using this to detect latin words
- Split image with padding for incliding accent
- Send image to Tesseract to convert to Meaningful String
- Compose all with postition info from OpenCV and String from Tesseract
- Generate to xml

