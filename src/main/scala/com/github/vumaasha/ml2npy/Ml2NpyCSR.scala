package com.github.vumaasha.ml2npy

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

  val compressionType: CompressionType = CompressionType.NONE
  val out = fs.create(file)
  val ml2NpyCSR = new Ml2NpyCSR

  override def write(key: Vector, value: Vector): Unit = {
    ml2NpyCSR.addRecord(key, value)
  }

  override def close(reporter: Reporter): Unit = {
    out.write(ml2NpyCSR.getBytes)
    out.close()

  }
}


object Ml2NpyCSRTester {
  def main(args: Array[String]): Unit = {
    /*
    To test the export start ipython and run following commands

    import numpy as np
    from scipy.sparse import csr_matrix
    loader = np.load('/tmp/test.npz')
    csr_matrix((loader['data'],loader['indices'],loader['indptr']),shape=loader['shape']).toarray()

    The imported matrix should contain the below values
    array([[ 0.        ,  0.1       ],
       [ 0.        ,  0.30000001],
       [ 0.5       ,  0.        ]], dtype=float32)
     */

    val csrGen = new Ml2NpyCSR
    val data: Seq[Vector] = Seq(
      new SparseVector(3, Array(0), Array(0.1)),
      new SparseVector(3, Array(1), Array(0.2)),
      new SparseVector(3, Array(2), Array(0.3))
    )
    val labels = Seq(
      new DenseVector(Array(0, 1)),
      new DenseVector(Array(1, 0)),
      new DenseVector(Array(1, 0))
    )
    data.zip(labels).foreach(tup => csrGen.addRecord(tup._1, tup._2))
    val fos = new FileOutputStream(new File("/tmp/data.npz"))
    fos.write(csrGen.getBytes)
    fos.close()
  }
}

