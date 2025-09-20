/*
 * Medical Waste Sorting Robot Controller
 * Arduino interface for controlling the rotating platform sorting mechanism
 * 
 * Features:
 * - Central rotating platform with multiple sorting bins
 * - Stepper motor control for precise positioning
 * - Servo-controlled waste item placement
 * - Safety sensors and emergency stop
 * - Communication with main processing unit
 */

#include <AccelStepper.h>
#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Pin definitions
#define STEP_PIN 2
#define DIR_PIN 3
#define ENABLE_PIN 4
#define SERVO_PIN 5
#define EMERGENCY_STOP_PIN 6
#define POSITION_SENSOR_PIN 7
#define WASTE_DETECT_PIN 8
#define LED_STATUS_PIN 9

// Sorting bin positions (in steps)
#define BIN_SHARPS 0
#define BIN_INFECTIOUS 512
#define BIN_PHARMACEUTICAL 1024
#define BIN_PATHOLOGICAL 1536
#define BIN_GENERAL 2048
#define HOME_POSITION 0

// Stepper motor configuration
#define STEPS_PER_REVOLUTION 2048
#define MAX_SPEED 1000
#define ACCELERATION 500

// Servo configuration
#define SERVO_PICKUP_ANGLE 90
#define SERVO_DROP_ANGLE 0
#define SERVO_HOME_ANGLE 45

// Initialize components
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
Servo wasteServo;
LiquidCrystal_I2C lcd(0x27, 16, 2);  // I2C address 0x27, 16x2 display

// Global variables
enum WasteCategory {
  SHARPS,
  INFECTIOUS,
  PHARMACEUTICAL,
  PATHOLOGICAL,
  GENERAL,
  UNKNOWN
};

enum SystemState {
  IDLE,
  ROTATING,
  PICKING_UP,
  DROPPING,
  HOMING,
  ERROR
};

WasteCategory currentWasteCategory = UNKNOWN;
SystemState currentState = IDLE;
bool emergencyStop = false;
bool wasteDetected = false;
long targetPosition = 0;
unsigned long lastActivity = 0;
unsigned long stateTimeout = 30000;  // 30 seconds timeout

// Waste category names for display
const char* wasteCategoryNames[] = {
  "SHARPS",
  "INFECTIOUS", 
  "PHARMACEUTICAL",
  "PATHOLOGICAL",
  "GENERAL",
  "UNKNOWN"
};

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("Medical Waste Sorting Robot - Starting...");
  
  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Waste Sorting");
  lcd.setCursor(0, 1);
  lcd.print("Robot Ready");
  
  // Initialize stepper motor
  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCELERATION);
  stepper.setEnablePin(ENABLE_PIN);
  stepper.enableOutputs();
  
  // Initialize servo
  wasteServo.attach(SERVO_PIN);
  wasteServo.write(SERVO_HOME_ANGLE);
  
  // Initialize pins
  pinMode(EMERGENCY_STOP_PIN, INPUT_PULLUP);
  pinMode(POSITION_SENSOR_PIN, INPUT_PULLUP);
  pinMode(WASTE_DETECT_PIN, INPUT);
  pinMode(LED_STATUS_PIN, OUTPUT);
  
  // Home the system
  homeSystem();
  
  Serial.println("System initialized successfully");
  updateDisplay("System Ready", "Waiting...");
}

void loop() {
  // Check emergency stop
  checkEmergencyStop();
  
  if (emergencyStop) {
    handleEmergencyStop();
    return;
  }
  
  // Check for serial commands
  handleSerialCommands();
  
  // Check waste detection
  checkWasteDetection();
  
  // Update state machine
  updateStateMachine();
  
  // Update display
  updateDisplayStatus();
  
  // Check for timeout
  checkTimeout();
  
  delay(100);  // Small delay for stability
}

void checkEmergencyStop() {
  emergencyStop = !digitalRead(EMERGENCY_STOP_PIN);
  if (emergencyStop) {
    digitalWrite(LED_STATUS_PIN, HIGH);
  } else {
    digitalWrite(LED_STATUS_PIN, LOW);
  }
}

void handleEmergencyStop() {
  stepper.stop();
  stepper.disableOutputs();
  wasteServo.write(SERVO_HOME_ANGLE);
  currentState = ERROR;
  updateDisplay("EMERGENCY STOP", "Press Reset");
}

void handleSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("SORT:")) {
      // Command format: SORT:category
      String category = command.substring(5);
      handleSortCommand(category);
    }
    else if (command == "HOME") {
      homeSystem();
    }
    else if (command == "STATUS") {
      sendStatus();
    }
    else if (command == "TEST") {
      runSelfTest();
    }
  }
}

void handleSortCommand(String category) {
  if (currentState != IDLE) {
    Serial.println("ERROR:System busy");
    return;
  }
  
  // Map category string to enum
  if (category == "SHARPS") {
    currentWasteCategory = SHARPS;
    targetPosition = BIN_SHARPS;
  }
  else if (category == "INFECTIOUS") {
    currentWasteCategory = INFECTIOUS;
    targetPosition = BIN_INFECTIOUS;
  }
  else if (category == "PHARMACEUTICAL") {
    currentWasteCategory = PHARMACEUTICAL;
    targetPosition = BIN_PHARMACEUTICAL;
  }
  else if (category == "PATHOLOGICAL") {
    currentWasteCategory = PATHOLOGICAL;
    targetPosition = BIN_PATHOLOGICAL;
  }
  else if (category == "GENERAL") {
    currentWasteCategory = GENERAL;
    targetPosition = BIN_GENERAL;
  }
  else {
    currentWasteCategory = UNKNOWN;
    targetPosition = BIN_GENERAL;  // Default to general waste
  }
  
  currentState = ROTATING;
  lastActivity = millis();
  
  Serial.print("INFO:Sorting to ");
  Serial.println(wasteCategoryNames[currentWasteCategory]);
}

void updateStateMachine() {
  switch (currentState) {
    case IDLE:
      // Wait for commands
      break;
      
    case ROTATING:
      handleRotatingState();
      break;
      
    case PICKING_UP:
      handlePickingUpState();
      break;
      
    case DROPPING:
      handleDroppingState();
      break;
      
    case HOMING:
      handleHomingState();
      break;
      
    case ERROR:
      // Error state - requires manual intervention
      break;
  }
}

void handleRotatingState() {
  if (!stepper.isRunning()) {
    if (stepper.distanceToGo() == 0) {
      // Reached target position
      currentState = PICKING_UP;
      lastActivity = millis();
      Serial.println("INFO:Position reached, starting pickup");
    } else {
      // Start moving to target
      stepper.moveTo(targetPosition);
    }
  }
}

void handlePickingUpState() {
  // Move servo to pickup position
  wasteServo.write(SERVO_PICKUP_ANGLE);
  delay(1000);  // Allow servo to reach position
  
  // Check if waste is actually picked up
  if (digitalRead(WASTE_DETECT_PIN)) {
    currentState = DROPPING;
    lastActivity = millis();
    Serial.println("INFO:Waste picked up, proceeding to drop");
  } else {
    // No waste detected, return to idle
    currentState = IDLE;
    Serial.println("WARNING:No waste detected for pickup");
  }
}

void handleDroppingState() {
  // Move servo to drop position
  wasteServo.write(SERVO_DROP_ANGLE);
  delay(2000);  // Allow waste to fall
  
  // Return servo to home position
  wasteServo.write(SERVO_HOME_ANGLE);
  delay(1000);
  
  // Return to home position
  currentState = HOMING;
  lastActivity = millis();
  Serial.println("INFO:Waste dropped, returning home");
}

void handleHomingState() {
  if (!stepper.isRunning()) {
    if (stepper.distanceToGo() == 0) {
      // Reached home position
      currentState = IDLE;
      Serial.println("INFO:Sorting cycle completed");
    } else {
      // Move to home position
      stepper.moveTo(HOME_POSITION);
    }
  }
}

void homeSystem() {
  currentState = HOMING;
  stepper.moveTo(HOME_POSITION);
  wasteServo.write(SERVO_HOME_ANGLE);
  lastActivity = millis();
  
  Serial.println("INFO:Homing system...");
}

void checkWasteDetection() {
  wasteDetected = digitalRead(WASTE_DETECT_PIN);
}

void updateDisplayStatus() {
  static unsigned long lastDisplayUpdate = 0;
  
  if (millis() - lastDisplayUpdate > 500) {  // Update every 500ms
    String line1 = "State: ";
    line1 += getStateString();
    
    String line2 = "Waste: ";
    line2 += wasteCategoryNames[currentWasteCategory];
    
    updateDisplay(line1, line2);
    lastDisplayUpdate = millis();
  }
}

void updateDisplay(String line1, String line2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
}

String getStateString() {
  switch (currentState) {
    case IDLE: return "IDLE";
    case ROTATING: return "ROTATING";
    case PICKING_UP: return "PICKUP";
    case DROPPING: return "DROPPING";
    case HOMING: return "HOMING";
    case ERROR: return "ERROR";
    default: return "UNKNOWN";
  }
}

void checkTimeout() {
  if (currentState != IDLE && currentState != ERROR) {
    if (millis() - lastActivity > stateTimeout) {
      Serial.println("ERROR:Operation timeout");
      currentState = ERROR;
      stepper.stop();
    }
  }
}

void sendStatus() {
  Serial.print("STATUS:");
  Serial.print("State=");
  Serial.print(getStateString());
  Serial.print(",Waste=");
  Serial.print(wasteCategoryNames[currentWasteCategory]);
  Serial.print(",Position=");
  Serial.print(stepper.currentPosition());
  Serial.print(",Emergency=");
  Serial.print(emergencyStop ? "TRUE" : "FALSE");
  Serial.print(",WasteDetected=");
  Serial.println(wasteDetected ? "TRUE" : "FALSE");
}

void runSelfTest() {
  Serial.println("INFO:Running self test...");
  
  // Test stepper motor
  stepper.moveTo(512);
  while (stepper.isRunning()) {
    stepper.run();
  }
  delay(1000);
  stepper.moveTo(0);
  while (stepper.isRunning()) {
    stepper.run();
  }
  
  // Test servo
  wasteServo.write(0);
  delay(1000);
  wasteServo.write(90);
  delay(1000);
  wasteServo.write(SERVO_HOME_ANGLE);
  
  // Test sensors
  bool emergencyState = digitalRead(EMERGENCY_STOP_PIN);
  bool positionState = digitalRead(POSITION_SENSOR_PIN);
  bool wasteState = digitalRead(WASTE_DETECT_PIN);
  
  Serial.print("TEST:Emergency=");
  Serial.print(emergencyState ? "OK" : "FAIL");
  Serial.print(",Position=");
  Serial.print(positionState ? "OK" : "FAIL");
  Serial.print(",Waste=");
  Serial.print(wasteState ? "DETECTED" : "NONE");
  Serial.println();
  
  Serial.println("INFO:Self test completed");
}
