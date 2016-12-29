package com.github.vumaasha.ml2npy

import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}

import scala.collection.mutable.ArrayBuffer


/**
  * Created by vumaasha on 24/12/16.
  */
abstract class NpyFile[V] {

  val magic: Array[Byte] = 0X93.toByte +: "NUMPY".getBytes(StandardCharsets.US_ASCII)
  val majorVersion = 1
  val minorVersion = 0
  val headerLenth = 70
  /*
  supported dtypes i2,i4,i8,f4,f8
   */
  val dtype: String
  val dataSize: Int
  val dataBuffer: ArrayBuffer[Byte] = ArrayBuffer.empty[Byte]
  var numElements:Int = 0

  def getBytes:Array[Byte] = {
    require(numElements >0, "Elements should be added to the buffer before getting bytes")
    (getHeader(numElements) ++ dataBuffer).toArray
  }

  def getHeader(dataLength: Int): ArrayBuffer[Byte] = {

    val header: ArrayBuffer[Byte] = ArrayBuffer()
    val description = s"{'descr': '$dtype', 'fortran_order': False, 'shape': ($dataLength,), }"

    val versionByteSize = 2
    val headerByteSize = 2
    val newLineSize = 1
    val unpaddedLength: Int = magic.length + versionByteSize + headerByteSize + description.length + newLineSize
    val isPaddingRequired = (unpaddedLength % 16) != 0
    val paddingLength = (((unpaddedLength / 16) + 1) * 16) - unpaddedLength
    val paddedDescription = if (isPaddingRequired) {
      description + (" " * paddingLength)
    } else description

    val headerLength: Int = unpaddedLength + paddingLength
    val content = ByteBuffer.allocate(headerLength)
      .order(ByteOrder.LITTLE_ENDIAN)
      .put(magic)
      .put(majorVersion.toByte)
      .put(minorVersion.toByte)
      .putShort(paddedDescription.length.toShort)
      .put(paddedDescription.getBytes(StandardCharsets.US_ASCII))

    header ++ content.array()
  }


  def addElements(data: Seq[V]) = {
    val dataLength = data.length.toString
    val description = s"{'descr': '$dtype', 'fortran_order': False, 'shape': ($dataLength,), }"

    val versionByteSize = 2
    val headerByteSize = 2
    val newLineSize = 1
    val unpaddedLength: Int = magic.length + versionByteSize + headerByteSize + description.length + newLineSize
    val isPaddingRequired = (unpaddedLength % 16) != 0
    val paddingLength = (((unpaddedLength / 16) + 1) * 16) - unpaddedLength
    val paddedDescription = if (isPaddingRequired) {
      description + (" " * paddingLength)
    } else description

    val headerLength: Int = unpaddedLength + paddingLength
    val content = ByteBuffer.allocate(headerLength + (data.length * dataSize))
      .order(ByteOrder.LITTLE_ENDIAN)
      .put(magic)
      .put(majorVersion.toByte)
      .put(minorVersion.toByte)
      .putShort(paddedDescription.length.toShort)
      .put(paddedDescription.getBytes(StandardCharsets.US_ASCII))

    data.foreach(addElement(content))
    content.flip
    content
  }

  protected def addElement(content: ByteBuffer)(elem: V): ByteBuffer

  def addToBuffer(elem:V):Unit = {
    val buffer: ByteBuffer = ByteBuffer.allocate(dataSize)
    dataBuffer ++= addElement(buffer)(elem).array()
    numElements += 1
  }
}

object NpyFile {

  implicit object IntNpyFile extends NpyFile[Int] {
    override val dtype: String = "<i4"
    override val dataSize: Int = 4

    override def addElement(content: ByteBuffer)(elem: Int) = content.putInt(elem)

  }

  implicit object LongNpyFile extends NpyFile[Long] {
    override val dtype: String = "<i8"
    override val dataSize: Int = 8

    override def addElement(content: ByteBuffer)(elem: Long): ByteBuffer = content.putLong(elem)
  }

  implicit object ShortNpyFile extends NpyFile[Short] {
    override val dtype: String = "<i2"
    override val dataSize: Int = 2

    override def addElement(content: ByteBuffer)(elem: Short) = content.putShort(elem)
  }

  implicit object ByteNpyFile extends NpyFile[Byte] {
    override val dtype: String = "<i1"
    override val dataSize: Int = 1

    override def addElement(content: ByteBuffer)(elem: Byte) = content.put(elem)
  }

  implicit object FloatNpyFile extends NpyFile[Float] {
    override val dtype: String = "<f4"
    override val dataSize: Int = 4

    override def addElement(content: ByteBuffer)(elem: Float) = content.putFloat(elem)
  }

  implicit object DoubleNpyFile extends NpyFile[Double] {
    override val dtype: String = "<f8"
    override val dataSize: Int = 8

    override def addElement(content: ByteBuffer)(elem: Double) = content.putDouble(elem)
  }

  def apply[V]()(implicit ev: NpyFile[V]): NpyFile[V] = ev
}

object NpyTester {
  def main(args: Array[String]): Unit = {
    /*
    To test the export start ipython and run following commands
    import numpy as np
    np.load('/tmp/test.npy')
    The imported matrix should contain the values (0.3,0.5)
     */

    val content = NpyFile[Float].addElements(Seq(0.3f, 0.5f))
    val channel = FileChannel.open(Paths.get("/tmp/test.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    channel.write(content)
    channel.close()

    val intContent = NpyFile[Int].addElements(1 to 10)
    val intChannel = FileChannel.open(Paths.get("/tmp/inttest.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel.write(intContent)
    intChannel.close()

  }
}