package com.indix.ml2npy.preprocessing

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, SparkSession}

trait DocGenerator[T] {

  def main(args: Array[String]): Unit = {

    val inputPath: String = args(0)
    val outputPath: String = args(1)
    val numParts: Int = args(2).toInt

    val spark = SparkSession.builder().appName("Ml2Npy").getOrCreate()
    val trainingRecords =readRecords(spark,inputPath)
    writeRecords(spark,trainingRecords,outputPath,pipeline,numParts)
  }

  def pipeline: Pipeline

  def readRecords(spark: SparkSession, inputPath: String):Dataset[T]

  def writeRecords(spark: SparkSession, sampledRecords: Dataset[T], outputPath: String, pipeline: Pipeline, numParts: Int)
}