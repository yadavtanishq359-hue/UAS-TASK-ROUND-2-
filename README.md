# UAS-TASK-ROUND-2

PROJECT LEARNINGS

The project focuses on detecting casualties and giving them priority order and making a planned map for rover to reach the casualties on the basis of severity score

The detected casualties are marked on the output image along with their rank, terrain level and coordinates

- Converting the image from BGR to HSV color space
- Using color masking to identify and remove unwanted regions.
- Detecting casualty shapes using contours.
- Identification the shape of each casualty:
  - Circle
  - Square
  - Star
- Identification the color of each casualty:
  - Red
  - Yellow
  - White
- Detection of the terrain level around each casualty.
- Calculating the coordinates of every detected casualty.
- Calculating the severity score using-
  Severity = Shape Score × Color Score

OUTPUT INFORMATION-
For every image the code detects casualty and give them ranking on the basis of severity order, gives coordinates of the casualty and terrain level of casualty
