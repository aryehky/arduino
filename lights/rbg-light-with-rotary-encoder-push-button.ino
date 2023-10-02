#include <BfButton.h>

int btnPin = 53;
int DT = 51;
int CLK = 49;
BfButton btn(BfButton::STANDALONE_DIGITAL, btnPin, true, LOW);

int PIN_RED = 26; 
int PIN_GREEN = 24;
int PIN_BLUE = 23;

int colorSelect = 0; // 0: red, 1: green, 2: blue
int redValue = 0;
int greenValue = 150;
int blueValue = 0;
int aState;
int aLastState;

void pressHandler (BfButton *btn, BfButton::press_pattern_t pattern) {
  switch (pattern) {
    case BfButton::SINGLE_PRESS:
      colorSelect = (colorSelect + 1) % 3;
      break;
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(CLK, INPUT_PULLUP);
  pinMode(DT, INPUT_PULLUP);
  aLastState = digitalRead(CLK);

  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BLUE, OUTPUT);

  btn.onPress(pressHandler);
  
}

void loop() {
  btn.read();
  
  aState = digitalRead(CLK);

  if (aState != aLastState) {
    if (digitalRead(DT) != aState) {
      changeColorValue(1);
    } else {
      changeColorValue(-1);
    }
  }
  
  aLastState = aState;

  analogWrite(PIN_RED, redValue);
  analogWrite(PIN_GREEN, greenValue);
  analogWrite(PIN_BLUE, blueValue);
}

void changeColorValue(int increment) {
  switch (colorSelect) {
    case 0:
      redValue = constrain(redValue + increment, 0, 255);
      break;
    case 1:
      greenValue = constrain(greenValue + increment, 0, 255);
      break;
    case 2:
      blueValue = constrain(blueValue + increment, 0, 255);
      break;
  }
  Serial.print("Red: "); Serial.print(redValue);
  Serial.print(" Green: "); Serial.print(greenValue);
  Serial.print(" Blue: "); Serial.println(blueValue);
}
