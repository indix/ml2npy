package com.indix.ml2npy

import java.io.{ByteArrayOutputStream, File, FileOutputStream}
import java.util.zip.{ZipEntry, ZipOutputStream}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.SequenceFile.CompressionType
import org.apache.hadoop.mapred.{RecordWriter, Reporter}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

/**
  * Created by vumaasha on 1/1/17.
  */
class Ml2NpyCSR {
  var rows: Int = 0
  var columns: Int = 0
  var numElements: Int = 0
  var numLabelValues: Int = 0
  var numLabelColumns: Int = 1
  val dataNpyFile = NpyFile[Float]
  val indicesNpyFile = NpyFile[Int]
  val indexPointersNpyFile = NpyFile[Int]
  indexPointersNpyFile.addToBuffer(numElements)

  val labelsNpyFile = NpyFile[Short]

  def addRecord(key: Vector, value: Vector) = {
    rows += 1
    val actives: Int = key.numActives
    numElements += actives
    columns = math.max(key.size, columns)
    val sparse: SparseVector = key.toSparse
    sparse.values.foreach(x => dataNpyFile.addToBuffer(x.toFloat))
    sparse.indices.foreach(x => indicesNpyFile.addToBuffer(x))
    indexPointersNpyFile.addToBuffer(numElements)

    numLabelColumns = math.max(numLabelColumns, value.numActives)
    numLabelValues += value.numActives
    value.toArray.map(_.toShort).foreach(labelsNpyFile.addToBuffer)
  }

  def getBytes: Array[Byte] = {
    def addEntry(zos: ZipOutputStream, file: String, bytes: Array[Byte]): Unit = {
      val entry = new ZipEntry(s"$file.npy")
      zos.putNextEntry(entry)
      zos.write(bytes)
      zos.closeEntry()
    }
    val bos = new ByteArrayOutputStream()
    val zos = new ZipOutputStream(bos)
    zos.setLevel(9)

    val dataShape = Array(rows, columns)
    val dataShapeB = NpyFile[Int].addElements(dataShape)

    val labelsShape = Array(rows, numLabelColumns)
    val labelsShapeB = NpyFile[Int].addElements(labelsShape)

    require(numLabelValues == (rows * numLabelColumns), "All the label vectors should be of same size")

    addEntry(zos, "data", dataNpyFile.getBytes)
    addEntry(zos, "indices", indicesNpyFile.getBytes)
    addEntry(zos, "indptr", indexPointersNpyFile.getBytes)
    addEntry(zos, "data_shape", dataShapeB.array())
    addEntry(zos, "labels", labelsNpyFile.getBytes)
    addEntry(zos, "labels_shape", labelsShapeB.array())
    zos.close()
    bos.toByteArray
  }

}

class Ml2NpyCSRWriter(fs: FileSystem, file: Path) extends RecordWriter[Vector, Vector] {

  val out = fs.create(file,true)
  val ml2NpyCSR = new Ml2NpyCSR

  override def write(key: Vector, value: Vector): Unit = {
    ml2NpyCSR.addRecord(key, value)
  }

  override def close(reporter: Reporter): Unit = {
    out.write(ml2NpyCSR.getBytes)
    out.close()

  }
}
}

