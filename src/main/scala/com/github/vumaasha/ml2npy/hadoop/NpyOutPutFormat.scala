package com.github.vumaasha.ml2npy.hadoop

import com.github.vumaasha.ml2npy.Ml2NpyCSRWriter
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.{FileOutputFormat, JobConf, RecordWriter}
import org.apache.hadoop.util.Progressable
import org.apache.spark.ml.linalg.Vector

/**
  * Created by vumaasha on 29/12/16.
  */
class NpyOutPutFormat extends FileOutputFormat[Vector, Vector] {

  override def getRecordWriter(ignored: FileSystem, job: JobConf, name: String, progress: Progressable): RecordWriter[Vector, Vector] = {

    val file = FileOutputFormat.getTaskOutputPath(job, name+".npz")
    val fs = file.getFileSystem(job)
    new Ml2NpyCSRWriter(fs,file)
  }
}
