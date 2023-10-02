
//Arduino Code - Rotary Encoder w push button
 
#include <BfButton.h>
 
int btnPin=53; //GPIO #3-Push button on encoder
int DT=51; //GPIO #4-DT on encoder (Output B)
int CLK=49; //GPIO #5-CLK on encoder (Output A)
BfButton btn(BfButton::STANDALONE_DIGITAL, btnPin, true, LOW);
 
int counter = 0;
int angle = 0; 
int aState;
int aLastState;  
 
//Button press hanlding function
void pressHandler (BfButton *btn, BfButton::press_pattern_t pattern) {
  switch (pattern) {
    case BfButton::SINGLE_PRESS:
      Serial.println("Single push");
      break;
      
    case BfButton::DOUBLE_PRESS:
      Serial.println("Double push");
      break;
      
    case BfButton::LONG_PRESS:
      Serial.println("Long push");
      break;
  }
}
 
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println(angle);
  pinMode(CLK,INPUT_PULLUP);
  pinMode(DT,INPUT_PULLUP);
  aLastState = digitalRead(CLK);
 
  //Button settings
  btn.onPress(pressHandler)
  .onDoublePress(pressHandler) // default timeout
  .onPressFor(pressHandler, 1000); // custom timeout for 1 second
}
 
void loop() {
  // put your main code here, to run repeatedly:
 
  //Wait for button press to execute commands
  btn.read();
  
  aState = digitalRead(CLK);
 
  //Encoder rotation tracking
  if (aState != aLastState){     
     if (digitalRead(DT) != aState) { 
       counter ++;
       angle ++;
     }
     else {
       counter--;
       angle --;
     }
     if (counter >=100 ) {
       counter =100;
     }
     if (counter <=-100 ) {
       counter =-100;
     }
     Serial.println(counter); 
  }   
  aLastState = aState;
}
