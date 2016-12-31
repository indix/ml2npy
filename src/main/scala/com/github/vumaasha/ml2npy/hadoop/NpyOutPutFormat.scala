package com.github.vumaasha.ml2npy.hadoop

import com.github.vumaasha.ml2npy.ml2npyCSRBuffer
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.hadoop.mapreduce.{RecordWriter, TaskAttemptContext}
import org.apache.spark.ml.linalg.Vector

/**
  * Created by vumaasha on 29/12/16.
  */
class NpyOutPutFormat extends FileOutputFormat[Vector, Vector] {
  override def getRecordWriter(job: TaskAttemptContext): RecordWriter[Vector, Vector] = {
    new ml2npyCSRBuffer
}

}
