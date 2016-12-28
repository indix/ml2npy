package com.github.vumaasha.ml2npy

import java.io.{ByteArrayOutputStream, File, FileOutputStream}
import java.nio.ByteBuffer
import java.util.zip.{ZipEntry, ZipOutputStream}

import org.apache.spark.ml.linalg.SparseVector

/**
  * Created by vumaasha on 25/12/16.
  */

class ml2npyCSR(data: Seq[Float], indices: Seq[Int], indexPointers: Seq[Int], rows: Int, columns: Int, labels: Seq[Short]*) {

  require(indexPointers.length -1 == rows)

  val dataB: ByteBuffer = (new FloatNpyFile).addElements(data)
  val indicesB: ByteBuffer = (new IntNpyFile).addElements(indices)
  val indexPointersB: ByteBuffer = (new IntNpyFile).addElements(indexPointers)
  val shapeB: ByteBuffer = (new IntNpyFile).addElements(Array(rows, columns))


  private def addEntry(zos: ZipOutputStream, file: String, bytes: Array[Byte]): Unit = {
    val entry = new ZipEntry(s"$file.npy")
    zos.putNextEntry(entry)
    zos.write(bytes)
    zos.closeEntry()
  }

  val zipOut = {
    val bos = new ByteArrayOutputStream()
    val zos = new ZipOutputStream(bos)
    def addLabels(labels: Seq[Short], index: Int): Unit = {
      require(labels.length == rows)
      val labelsB = (new ShortNpyFile).addElements(labels)
      addEntry(zos, s"label_$index", labelsB.array())
    }
    addEntry(zos, "data", dataB.array())
    addEntry(zos, "indices", indicesB.array())
    addEntry(zos, "indptr", indexPointersB.array())
    addEntry(zos, "shape", shapeB.array())
    labels.zipWithIndex.foreach(tup => addLabels(tup._1, tup._2))
    zos.close()
    bos
  }

}

object ml2npyCSR {
  def apply(data: Seq[Float], indices: Seq[Int], indexPointers: Seq[Int], rows: Int, columns: Int, labels: Seq[Short]*): ml2npyCSR = new ml2npyCSR(data, indices, indexPointers, rows, columns, labels:_*)

  def apply(vectors: Seq[SparseVector], labels: Seq[Short]*): ml2npyCSR = {
    val indices = vectors.flatMap(_.indices)
    val values: Seq[Float] = vectors.flatMap(_.values).map(_.toFloat)
    val indPtr = vectors.map(_.numActives).toArray.scanLeft(0)(_ + _)
    val rows = vectors.length
    val columns = vectors.map(_.size).max
    new ml2npyCSR(values, indices, indPtr, rows, columns,labels:_*)
  }
}

object ml2npyCSRTester {
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

    val csr = new ml2npyCSR(Seq(0.1f, 0.3f, 0.5f), Seq(1, 1, 0), Seq(0, 1, 2, 3), 3, 2)
    val fos = new FileOutputStream(new File("/tmp/data.npz"))
    csr.zipOut.writeTo(fos)
    csr.zipOut.close()
    fos.close()
  }
}

object SparseVectorTester {
  def main(args: Array[String]): Unit = {
    val x = Seq(new SparseVector(4, Array(0, 1), Array(0.3, 0.5)), new SparseVector(4, Array(1, 3), Array(0.3, 0.5)), new SparseVector(4, Array(1, 2), Array(0.3, 0.5)))
    val labels = Seq(1.toShort,2.toShort,0.toShort)
    val csr = ml2npyCSR(x,labels,labels)
    val fos = new FileOutputStream(new File("/tmp/csr.npz"))
    csr.zipOut.writeTo(fos)
    csr.zipOut.close()
    fos.close()

  }
}
