# lightnet-ml
A light weight machine learning library written in Qt/C++

# Requirements

## Tools
1. Qt SDK
2. C++ compiler (Qt SDK usually ships with the MinGW 32-bit default compiler)

# Setup
1. Clone the repository or download it as a zip file.
2. Unzip the file to a directory of your liking.
3. Open up Qt Creator and navigate to the directory above and open the **lightnet.pro** file.
4. This should open up the project, and should present you with a view similar to this one (Expand the folders for full view).![image](https://user-images.githubusercontent.com/37830837/119785251-b478e500-bed7-11eb-93d6-652f34050e25.png)

5. The **main.cpp** file contains a sample code that illustrates how to use the library.![image](https://user-images.githubusercontent.com/37830837/119785715-22251100-bed8-11eb-8faa-d9f685907813.png)

6. In the project folder is a **data** directory which contains some sample datasets and a trained model as shown. Feel free to explore them..![image](https://user-images.githubusercontent.com/37830837/119786045-729c6e80-bed8-11eb-9f85-278003eb9f06.png)

7. The **src** folder contains the core library classes for the neural network.![image](https://user-images.githubusercontent.com/37830837/119786327-c14a0880-bed8-11eb-8f3a-41b5e9202ec3.png)

# Sample usage
We are going to illustrate how we can use this library to train a model. For this illustration, we a going to use the **iris_flowers.csv** dataset.

Proceed as follows:
1. Clear the main.cpp file so that you remain with the main function as follows:
    
    ```
    #include <iostream>

    using namespace std;
    using namespace LightNet;

    int main()
    {
        return 0;
    }

    ```
2. Create a function prototype called **void trainAndSave()** as shown below:
    
    ```
    #include <iostream>

    using namespace std;
    using namespace LightNet;

    void trainAndSave();
    
    int main()
    {
        return 0;
    }

    ```
3. Next create the function definition below the main function like this:

    ```
    #include <iostream>

    using namespace std;
    using namespace LightNet;

    void trainAndSave();
    
    int main()
    {
        return 0;
    }
    
    void trainAndSave(){
    
    }

    ```
4. We are now going to import our data set using the Dataset class as shown below (Add this include statement **#include "src/dataset.h"**). Additionally, we are going to scale our data for better performance. (The scaling method used is **minmax scaling**).

    ```
    #include <iostream>
    
    #include "src/dataset.h"

    using namespace std;
    using namespace LightNet;

    void trainAndSave();
    
    int main()
    {
        return 0;
    }
    
    void trainAndSave(){
        Dataset dataset("D:/GitHub/lightnet-ml/data/iris_flowers.csv", true);
        dataset.scale();
    }

    ```
As you may have noted, my dataset is stored in my computer at the path **"D:/GitHub/lightnet-ml/data/iris_flowers.csv"**. Please replace this with the path where your data set is stored.
5. The next thing to do is to split the data into traing and testing data. We do so using the **splitTestData** function, which takes a percentage value, representing the percentage of the data that will be used for testing. In our case, we use **5**, which means that 5 percent of the data will be allocated for testing. The function then returns the split data from the original dataset. This means that the original dataset reduces in size die to the data split from it.

    ```
    #include <iostream>
    
    #include "src/dataset.h"

    using namespace std;
    using namespace LightNet;

    void trainAndSave();
    
    int main()
    {
        return 0;
    }
    
    void trainAndSave(){
        Dataset dataset("D:/GitHub/lightnet-ml/data/iris_flowers.csv", true);
        dataset.scale();
        
        Dataset testData = dataset.splitTestData(5);
    }

    ```
    
6. Now that we have our data ready, we proceed to build the architecture of our neural network (Add this include statement **#include "src/nnclassifier.h"**). The first argument it takes is the architecture, defined with a vector. The length of the vector defines the number of layers the neuron has. For example the vector **{5, 10, 4}** defines a neural network of 3 layers, with 5, 10 and 4 neurons respectively. The number of neurons in the first layer must be equivalent to the number of inputs in the data set. That is one neuron per input. To achieve this, we use the **getInputCount** function. Similarly, the number of neurons in the last layer must be equivalent to the number of unique targets in the data set. In a similar way we use the **getUniqueTargetCount** function to make sure of this. The number of layers in the network as well as the number of neurons in the hidden layers can vary depending on which architecture works well. Its an issue of trial and error.

    ```
    #include <iostream>
    
    #include "src/dataset.h"
    #include "src/nnclassifier.h"

    using namespace std;
    using namespace LightNet;

    void trainAndSave();
    
    int main()
    {
        return 0;
    }
    
    void trainAndSave(){
        Dataset dataset("D:/GitHub/lightnet-ml/data/iris_flowers.csv", true);
        dataset.scale();
        
        Dataset testData = dataset.splitTestData(5);
        
        NNClassifier net({dataset.getInputCount(), 10, dataset.getUniqueTargetCount()}, dataset);
    }

    ```
