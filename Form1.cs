using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace opencv6
{
    public partial class Form1 : Form
    {
        private string _refImagesFolderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "参考图片");
        private string _resultFolderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "识别结果");
        private string _featureDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "featureData.bin");
        public Form1()
        {
            InitializeComponent();
            if (!Directory.Exists(_refImagesFolderPath))
                Directory.CreateDirectory(_refImagesFolderPath);
            if (!Directory.Exists(_resultFolderPath))
                Directory.CreateDirectory(_resultFolderPath);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "图片文件|*.jpg;*.jpeg;*.png;";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                string fileName = openFileDialog.FileName;
                pictureBox1.Image = Image.FromFile(fileName);

                Image<Bgr, byte> inputImage = new Image<Bgr, byte>(fileName);
                Image<Bgr, byte> processedImage = PreprocessImage(inputImage);

                // 修改此行，使用ToBitmap()方法将Image<Bgr, byte>转换为Bitmap
                pictureBox2.Image = processedImage.ToBitmap();

                string matchedRefImageName = MatchReferenceImage(processedImage);

                if (!string.IsNullOrEmpty(matchedRefImageName))
                {
                    textBox1.Text = matchedRefImageName;
                    pictureBox3.Image = Image.FromFile(Path.Combine(_refImagesFolderPath, matchedRefImageName));
                }
                else
                {
                    textBox1.Text = "未识别";
                    pictureBox3.Image = null;
                }

                string resultFileName = $"识别结果_{DateTime.Now.ToString("yyyyMMddHHmmss")}.jpg";
                processedImage.Save(Path.Combine(_resultFolderPath, resultFileName));
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            GenerateReferenceFeatureData();
        }
        private Image<Bgr, byte> PreprocessImage(Image<Bgr, byte> inputImage)
        {
            // 缩放图像
            int height = 1000;
            int width = (int)((float)inputImage.Width / inputImage.Height * height);
            Image<Bgr, byte> resizedImage = inputImage.Resize(width, height, Emgu.CV.CvEnum.Inter.Linear);

            // 裁剪图像
            Rectangle roi = new Rectangle(200, 100, 800, 800);
            resizedImage.ROI = roi;
            Image<Bgr, byte> croppedImage = resizedImage.Copy();

            // 边缘检测
            Image<Gray, byte> grayImage = croppedImage.Convert<Gray, byte>();
            Image<Gray, byte> edgeImage = grayImage.Canny(100, 200);
            Image<Bgr, byte> edgeOverlayImage = croppedImage.Copy();
            edgeOverlayImage.SetValue(new Bgr(0, 255, 0), edgeImage);

            // 霍夫圆变换
            CircleF[] circles = CvInvoke.HoughCircles(edgeImage, Emgu.CV.CvEnum.HoughModes.Gradient, 1, 100, 100, 50, 300, 400);
            Image<Bgr, byte> houghCircleImage = edgeOverlayImage.Copy();
            foreach (CircleF circle in circles)
            {
                houghCircleImage.Draw(circle, new Bgr(255, 0, 0), 2);
            }

            // 裁剪并遮罩圆外部
            if (circles.Length > 0)
            {
                CircleF largestCircle = circles.OrderByDescending(c => c.Radius).First();
                Image<Bgr, byte> maskedImage = houghCircleImage.CopyBlank();
                for (int y = 0; y < maskedImage.Height; y++)
                {
                    for (int x = 0; x < maskedImage.Width; x++)
                    {
                        float dist = (x - largestCircle.Center.X) * (x - largestCircle.Center.X) + (y - largestCircle.Center.Y) * (y - largestCircle.Center.Y);
                        if (dist < largestCircle.Radius * largestCircle.Radius)
                        {
                            maskedImage.Data[y, x, 0] = houghCircleImage.Data[y, x, 0];
                            maskedImage.Data[y, x, 1] = houghCircleImage.Data[y, x, 1];
                            maskedImage.Data[y, x, 2] = houghCircleImage.Data[y, x, 2];
                        }
                        else
                        {
                            maskedImage.Data[y, x, 0] = 255;
                            maskedImage.Data[y, x, 1] = 255;
                            maskedImage.Data[y, x, 2] = 255;
                        }
                    }
                }
                return maskedImage;
            }
            else
            {
                return houghCircleImage;
            }
        }

        private string MatchReferenceImage(Image<Bgr, byte> processedImage)
        {
            if (!File.Exists(_featureDataPath))
                GenerateReferenceFeatureData();

            using (FileStream fileStream = new FileStream(_featureDataPath, FileMode.Open))
            {
                using (BinaryReader reader = new BinaryReader(fileStream))
                {
                    int refImageCount = reader.ReadInt32();
                    ORBDetector detector = new ORBDetector();
                    VectorOfKeyPoint queryKeypoints = new VectorOfKeyPoint();
                    Mat queryDescriptors = new Mat();
                    detector.DetectAndCompute(processedImage, null, queryKeypoints, queryDescriptors, false);

                    BFMatcher matcher = new BFMatcher(DistanceType.Hamming);
                    int bestMatchCount = 0;
                    string bestMatchImageName = string.Empty;

                    for (int i = 0; i < refImageCount; i++)
                    {
                        string imageName = reader.ReadString();
                        int keypointCount = reader.ReadInt32();
                        List<MKeyPoint> refKeypointsList = new List<MKeyPoint>(keypointCount);

                        for (int j = 0; j < keypointCount; j++)
                        {
                            MKeyPoint keypoint = new MKeyPoint
                            {
                                Point = new PointF(reader.ReadSingle(), reader.ReadSingle()),
                                Size = reader.ReadSingle(),
                                Angle = reader.ReadSingle(),
                                Response = reader.ReadSingle(),
                                Octave = reader.ReadInt32(),
                                ClassId = reader.ReadInt32()
                            };
                            refKeypointsList.Add(keypoint);
                        }

                        VectorOfKeyPoint refKeypoints = new VectorOfKeyPoint(refKeypointsList.ToArray());

                        Mat refDescriptors = new Mat();
                        int rows = reader.ReadInt32();
                        int cols = reader.ReadInt32();
                        int ElementSize = reader.ReadInt32();



                        //byte[] descriptorData = reader.ReadBytes(rows * cols);
                        byte[] descriptorData = null;
                        int count = rows * cols;

                        if (count >= 0)
                        {
                            descriptorData = reader.ReadBytes(count);
                        }
                        else
                        {
                            MessageBox.Show(rows + "-" + cols);
                            // 在这里处理错误情况，例如显示错误消息或跳过读取此条记录
                        }

                        refDescriptors.Create(rows, cols, (Emgu.CV.CvEnum.DepthType)ElementSize, 1);
                        refDescriptors.SetTo<byte>(descriptorData);

                        VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();

                        matcher.KnnMatch(queryDescriptors, refDescriptors, matches, 2);


                        int goodMatchCount = 0;

                        for (int k = 0; k < matches.Size; k++)
                        {
                            MDMatch[] match = matches[k].ToArray();
                            if (match[0].Distance < 0.75 * match[1].Distance)
                            {
                                goodMatchCount++;
                            }
                        }

                        if (goodMatchCount > bestMatchCount)
                        {
                            bestMatchCount = goodMatchCount;
                            bestMatchImageName = imageName;
                        }
                    }

                    return bestMatchImageName;
                }
            }
        }

        private void GenerateReferenceFeatureData()
        {
            using (FileStream fileStream = new FileStream(_featureDataPath, FileMode.Create))
            {
                using (BinaryWriter writer = new BinaryWriter(fileStream))
                {
                    DirectoryInfo directoryInfo = new DirectoryInfo(_refImagesFolderPath);
                    FileInfo[] referenceImageFiles = directoryInfo.GetFiles("*.jpg");
                    writer.Write(referenceImageFiles.Length);

                    ORBDetector detector = new ORBDetector();

                    foreach (FileInfo refImageFile in referenceImageFiles)
                    {
                        Image<Bgr, byte> refImage = new Image<Bgr, byte>(refImageFile.FullName);
                        VectorOfKeyPoint refKeypoints = new VectorOfKeyPoint();
                        Mat refDescriptors = new Mat();
                        detector.DetectAndCompute(refImage, null, refKeypoints, refDescriptors, false);

                        writer.Write(refImageFile.Name);
                        writer.Write(refKeypoints.Size);

                        for (int i = 0; i < refKeypoints.Size; i++)
                        {
                            MKeyPoint keypoint = refKeypoints[i];
                            writer.Write(keypoint.Point.X);
                            writer.Write(keypoint.Point.Y);
                            writer.Write(keypoint.Size);
                            writer.Write(keypoint.Angle);
                            writer.Write(keypoint.Response);
                            writer.Write(keypoint.Octave);
                            writer.Write(keypoint.ClassId);
                        }
                        //MessageBox.Show(refKeypoints.Size.ToString());
                        byte[] data = new byte[refDescriptors.Rows * refDescriptors.Cols * refDescriptors.ElementSize];
                        //MessageBox.Show(refDescriptors.Rows.ToString());
                        //MessageBox.Show(refDescriptors.Cols.ToString());
                        //MessageBox.Show(refDescriptors.ElementSize.ToString());
                        Marshal.Copy(refDescriptors.DataPointer, data, 0, data.Length);
                        writer.Write(refDescriptors.Rows);
                        writer.Write(refDescriptors.Cols);
                        writer.Write(refDescriptors.ElementSize);
                        writer.Write(data.Length);
                        writer.Write(data);
                    }
                }
            }
        }
    }
}
