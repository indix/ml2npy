package com.github.vumaasha.ml2npy

import java.io.{ByteArrayOutputStream, File, FileOutputStream}
import java.nio.ByteBuffer
import java.util.zip.{ZipEntry, ZipOutputStream}

/**
  * Created by vumaasha on 25/12/16.
  */

class ml2npyCSR(data: Seq[Float], indices: Seq[Int], indexPointers: Seq[Int], rows: Int, columns: Int) {
  val dataB: ByteBuffer = (new FloatNpyFile).addElements(data)
  val indicesB: ByteBuffer = (new IntNpyFile).addElements(indices)
  val indexPointersB: ByteBuffer = (new IntNpyFile).addElements(indexPointers)
  val shapeB: ByteBuffer = (new IntNpyFile).addElements(Array(rows,columns))

  val zipOut = {
    val bos = new ByteArrayOutputStream()
    val zos = new ZipOutputStream(bos)
    def addEntry(file:String,bytes:Array[Byte]): Unit = {
      val entry = new ZipEntry(s"$file.npy")
      zos.putNextEntry(entry)
      zos.write(bytes)
      zos.closeEntry()
    }
    addEntry("data",dataB.array())
    addEntry("indices",indicesB.array())
    addEntry("indxptr",indexPointersB.array())
    addEntry("shape",shapeB.array())
    zos.close()
    bos
  }

}

object ml2npyCSRTester {
  def main(args: Array[String]): Unit = {
    /*
    To test the export start ipython and run following commands

    import numpy as np
    from scipy.sparse import csr_matrix
    loader = np.load('/tmp/test.npz')
    csr_matrix((loader['data'],loader['indices'],loader['indxptr']),shape=loader['shape']).toarray()

    The imported matrix should contain the below values
    array([[ 0.        ,  0.1       ],
       [ 0.        ,  0.30000001],
       [ 0.5       ,  0.        ]], dtype=float32)
     */

    val csr = new ml2npyCSR(Seq(0.1f,0.3f,0.5f),Seq(1,1,0),Seq(0,1,2,3),3,2)
    val fos = new FileOutputStream(new File("/tmp/data.npz"))
    csr.zipOut.writeTo(fos)
    csr.zipOut.close()
    fos.close()
  }
}
