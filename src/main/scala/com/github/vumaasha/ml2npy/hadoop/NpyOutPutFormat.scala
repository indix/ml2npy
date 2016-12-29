package com.github.vumaasha.ml2npy.hadoop

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.SequenceFile.CompressionType
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.hadoop.mapred.{FileOutputFormat, JobConf, RecordWriter}
import org.apache.hadoop.util.Progressable
import org.apache.spark.ml.linalg.Vector

/**
  * Created by vumaasha on 29/12/16.
  */
/*class NpyOutPutFormat extends FileOutputFormat[Vector,Vector]{
  override def getRecordWriter(ignored: FileSystem, job: JobConf, name: String, progress: Progressable): RecordWriter[BytesWritable, NullWritable] = {
    val file:Path = FileOutputFormat.getTaskOutputPath(job,name)
    val fs:FileSystem = file.getFileSystem(job)
    val compressionType:CompressionType = CompressionType.NONE
  }
}*/
