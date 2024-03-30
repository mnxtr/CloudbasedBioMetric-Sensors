# CSE498R Directed Research | Cloud-based Biometric Sensor


[Tasks](https://www.notion.so/ab800a73cf014937b95d28b0d7752da7)

[Task list](https://www.notion.so/ab800a73cf014937b95d28b0d7752da7)

To verify that the heart rate monitor is working as expected, [open the serial monitor](https://learn.sparkfun.com/tutorials/terminal-basics/arduino-serial-monitor-windows-mac-linux)
 at **9600 baud**

## interfacing with arduino and AD8232

### **/src/sketch.ino**

```cpp
#include <Wire.h>
#include <AD8232.h>

// Define the AD8232 module
AD8232 heartRate;

void setup()
{
  Serial.begin(9600);   // Initialize serial communication
  heartRate.begin();    // Initialize the AD8232 module
}

void loop()
{
  float heart_rate = heartRate.getHeartRate();  // Read the heart rate value
  Serial.println("Heart rate: " + String(heart_rate) + " bpm");  // Print the heart rate value
  delay(1000);  // Wait for 1 second
}
```

. You should see values printed on the screen. Below is an example output with the sensors connected to the forearms and right leg. Your serial work should spike between +300/-200 around the center value of about ~500.

## List of hardware required

- Arduino Rev3 atmega328  + Wi-Fi  → as MCU
- [AD8232 Heart Sensing Monitor](https://www.analog.com/en/products/ad8232.html)
- Breadboard
- Breadboard connector
- Electrode Sensor Surface pad → Disposable
- Pulse Sensor Module.
- `EEG Muscle Sensor.`

---

## **Software**

- Arduino IDE
- [Arduino Cloud - native](https://cloud.arduino.cc/home/?get-started=true)
- Fritzing
- [Excali Draw](https://excalidraw.com/)

 

---

> Serial Plotter
> 

![Initial Prototype](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/00037f21-75d7-4f20-9080-6f4012dd424c/296935907_564970585168162_8227733148751023666_n.png)

Initial Prototype

![First Test Case Requires Finetuning](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/48fc0350-1b47-46ba-8340-343493999089/274212130_436201271632733_8440882447172100438_n.png)

First Test Case Requires Finetuning

## Arduino Rev3 atmega328 + AD8232 Connection.

![Wiring ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f1a9496a-8597-4978-8c61-0a3c9d8c5615/Untitled.jpeg)

Wiring 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/080bc427-135b-4308-bc45-9b9b6b1c4cd4/Untitled.png)

![HeartRate_Normal.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/90b1ef18-734e-4b93-aa98-ad7b6df99b26/HeartRate_Normal.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/708e4173-88c1-41b7-b976-b6efcb2ed370/Untitled.png)

- 
- Cable Color Signal **Black** RA (Right Arm ) | **Blue** LA (Left Arm) | **Red** RL (Right Leg)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c8aa6b52-07e1-4816-abab-531831b076a7/Untitled.png)

![Enhancement+](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/92daf233-71b4-48c7-ab0b-f60795f85a5e/Untitled.png)

Enhancement+

Arduino layout with 7 Segment Led 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2284e2e0-9bc4-458a-9863-9e2b902f71fa/Untitled.png)


Research design and problem addressing

-----------------


- A research problem is a specific issue, contradiction, or gap in existing knowledge that needs to be addressed.

- Identifying a research problem involves recognizing something problematic that requires attention, explanation, or a solution.

- It is essential to have a well-defined research problem to guide the research process and to contribute to the field of study.

- The process of defining a research problem includes:

  - Identifying a broad topic of interest.

  - Conducting a literature review to understand the current state of knowledge.
  - Narrowing down to a specific issue that is under-explored or controversial.
  - Formulating the problem into a clear, concise statement or question.
- Types of research problems include:
  - Descriptive: Documenting certain phenomena or situations.
  - Exploratory: Investigating a topic to generate new insights or hypotheses.
  - Explanatory: Understanding the causes or effects of certain phenomena.
  - Predictive: Anticipating future occurrences based on current trends or data.
  - Evaluative: Assessing the effectiveness of interventions, programs, or policies.
- Defining a research problem is crucial for:
  - Setting the direction and scope of the study.
  - Ensuring the research is focused and manageable.
  - Avoiding duplication of existing knowledge.
  - Making a meaningful contribution to the academic field or practical application.
- For more detailed guidance on defining a research problem, resources such as Scribbr's articles can provide step-by-step instructions and examples.


## Scopes





