//week 1: the system can collect data repeatedly and output the data by serial port. Still need to copy from serial monitor. 
//week 2 update: add a FIR filter
//plan: the system is expected to automatically generate an excel file.
void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);
pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  int i = 0;
  int j = 0;
  int sum_1; 
  int sum_2;
  int avgNumber;
  int sampleNumber = 500;                  //单通道每组数据的个数
  int sensorValue_01[sampleNumber] = {0};
  int sensorValue_02[sampleNumber] = {0};  //以上两个数组用于存储采集到的信号
  int sensorValue_01_FIR[sampleNumber] = {0};
  int sensorValue_02_FIR[sampleNumber] = {0};  //以上两个数组用于存储降噪处理过的信号
  digitalWrite(LED_BUILTIN, HIGH);

  
  for(i = 0; i < sampleNumber; i++)
  {
    sensorValue_01[i] = analogRead(A0);
    sensorValue_02[i] = analogRead(A1);
    delay(5); 
  }

   digitalWrite(LED_BUILTIN, LOW);
  // 使用FIR滤波器进行降噪处理
  avgNumber = 10; //FIR滤波器的n参数
  for(i = 0; i <= sampleNumber - avgNumber; i++)
  {
    sum_1 = 0;
    sum_2 = 0;
    for(j = 0; j < avgNumber; j++)
    {
      sum_1 = sum_1 + sensorValue_01[i + j];
      sum_2 = sum_2 + sensorValue_02[i + j];
    }
    //----------------------------------------------------------------------------------------
    //位于中间的点用均值代替
    if(avgNumber % 2 == 0)  //若avgNumber为偶数，则取中间**两个数**由均值代替
    {
       sensorValue_01_FIR[i + avgNumber/2 - 1] = sum_1 / avgNumber;
       sensorValue_01_FIR[i + avgNumber/2] = sum_1 / avgNumber;
       sensorValue_02_FIR[i + avgNumber/2 - 1] = sum_2 / avgNumber;
       sensorValue_02_FIR[i + avgNumber/2] = sum_2 / avgNumber;
    }
    else //若avgNumber为奇数，则取中间**一个数**由均值代替
    {
      sensorValue_01_FIR[i + (avgNumber - 1)/2] = sum_1 / avgNumber;
      sensorValue_02_FIR[i + (avgNumber - 1)/2] = sum_2 / avgNumber; 
    }
    
    //----------------------------------------------------------------------------------------
    // 补全数组开始和末尾缺失的数值
    if(avgNumber % 2 == 0)  //若avgNumber为偶数，需要补全编号为0 ~ {avgNumber/2-2}以及499-{avgNumber/2-2} ~ 499的数值
    {
      if(i == 0)
      {
         for(j = 0; j <= avgNumber/2-2; j++)
        {
          sensorValue_01_FIR[j] = sum_1 / avgNumber;
          sensorValue_02_FIR[j] = sum_2 / avgNumber;
        }
      }
      if(i == sampleNumber - avgNumber)
      {
        for(j = sampleNumber + 1 - avgNumber/2; j <= sampleNumber - 1; j++)
        {
          sensorValue_01_FIR[j] = sum_1 / avgNumber;
          sensorValue_02_FIR[j] = sum_2 / avgNumber;
        }
      }
       
    }
    else  //若avgNumber为奇数，需要补全编号为0 ~ {(avgNumber-1)/2 - 1}以及499 - {(avgNumber-1)/2 - 1} ~ 499的数值
    {
      if(i == 0)
      {
         for(j = 0; j <= (avgNumber-1)/2 - 1; j++)
        {
          sensorValue_01_FIR[j] = sum_1 / avgNumber;
          sensorValue_02_FIR[j] = sum_2 / avgNumber;
        }
      }
      if(i == sampleNumber - avgNumber)
      {
        for(j = sampleNumber - (avgNumber-1)/2; j <= sampleNumber - 1; j++)
        {
          sensorValue_01_FIR[j] = sum_1 / avgNumber;
          sensorValue_02_FIR[j] = sum_2 / avgNumber;
        }
      }
    }
    }
  for(i = 0; i<sampleNumber; i++)
  {
    Serial.println(sensorValue_01_FIR[i]);
  }
 // Serial.println("**********end with 01**********");
   for(i = 0; i<sampleNumber; i++)
  {
    Serial.println(sensorValue_02_FIR[i]);
  }
 // Serial.println("**********end with 02**********");
  
   delay(1500);
}
