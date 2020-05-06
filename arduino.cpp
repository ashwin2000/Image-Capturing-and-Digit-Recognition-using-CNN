#code for arduino board to capture images with ultrasonic sensor

int trigger_pin = 13;
int echo_pin = 11; 
float time_taken;


void setup() { 
Serial.begin(9600); 
pinMode(trigger_pin, OUTPUT); 
pinMode(echo_pin, INPUT); 
}

void loop() {

digitalWrite(trigger_pin, LOW); 
delayMicroseconds(2000); 
digitalWrite(trigger_pin, HIGH);
delayMicroseconds(10); 
digitalWrite(trigger_pin, LOW);
time_taken = pulseIn(echo_pin, HIGH); 
Serial.println(time_taken); 
delay(50);

}
