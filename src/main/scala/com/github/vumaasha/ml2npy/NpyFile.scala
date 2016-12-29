package com.github.vumaasha.ml2npy

import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}

import scala.collection.immutable.Range.Inclusive
import scala.collection.mutable.ArrayBuffer


/**
  * Created by vumaasha on 24/12/16.
  */
abstract class NpyFile[V] {

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

    val magic: Array[Byte] = 0X93.toByte +: "NUMPY".getBytes(StandardCharsets.US_ASCII)
    val majorVersion = 1
    val minorVersion = 0

    val header: ArrayBuffer[Byte] = ArrayBuffer.empty[Byte]
    val description = s"{'descr': '$dtype', 'fortran_order': False, 'shape': ($dataLength,), }"

    val versionByteSize = 2
    val headerByteSize = 2
    val newLineSize = 1
    val unpaddedLength: Int = magic.length + versionByteSize + headerByteSize + description.length + newLineSize
    val isPaddingRequired = (unpaddedLength % 16) != 0
    val paddingLength = 16 - (unpaddedLength % 16)
    val paddedDescription = {
      if (isPaddingRequired) {
        description + (" " * paddingLength)
      } else description
    } + "\n"

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
    val header = getHeader(data.length)
    val headerLength: Int = header.length
    val content = ByteBuffer.allocate(headerLength + (data.length * dataSize))
      .order(ByteOrder.LITTLE_ENDIAN)
      .put(header.toArray)

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
  class IntNpyFile extends NpyFile[Int] {
    override val dtype: String = "<i4"
    override val dataSize: Int = 4

    override def addElement(content: ByteBuffer)(elem: Int) = content.putInt(elem)
  }

  implicit object IntNpyFile extends IntNpyFile {
    def apply() = new IntNpyFile()
  }

  class LongNpyFile extends NpyFile[Long] {
    override val dtype: String = "<i8"
    override val dataSize: Int = 8

    override def addElement(content: ByteBuffer)(elem: Long): ByteBuffer = content.putLong(elem)
  }

  implicit object LongNpyFile extends LongNpyFile {
    def apply() = new LongNpyFile()
  }

  class ShortNpyFile extends NpyFile[Short] {
    override val dtype: String = "<i2"
    override val dataSize: Int = 2

    override def addElement(content: ByteBuffer)(elem: Short) = content.putShort(elem)
  }

  implicit object ShortNpyFile extends ShortNpyFile {
    def apply() = new ShortNpyFile()
  }

  class ByteNpyFile extends NpyFile[Byte] {
    override val dtype: String = "<i1"
    override val dataSize: Int = 1

    override def addElement(content: ByteBuffer)(elem: Byte) = content.put(elem)
  }

  implicit object ByteNpyFile extends ByteNpyFile {
    def apply() = new ByteNpyFile()
  }

  class FloatNpyFile extends NpyFile[Float] {
    override val dtype: String = "<f4"
    override val dataSize: Int = 4

    override def addElement(content: ByteBuffer)(elem: Float) = content.putFloat(elem)
  }

  implicit object FloatNpyFile extends FloatNpyFile {
    def apply() = new FloatNpyFile()
  }

  class DoubleNpyFile extends NpyFile[Double] {
    override val dtype: String = "<f8"
    override val dataSize: Int = 8

    override def addElement(content: ByteBuffer)(elem: Double) = content.putDouble(elem)
  }

  implicit object DoubleNpyFile extends DoubleNpyFile {
    def apply() = new DoubleNpyFile()
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

    val intContent2 = NpyFile[Int].addElements(2 to 20)
    val intChannel2 = FileChannel.open(Paths.get("/tmp/inttest2.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel2.write(intContent2)
    intChannel2.close()

    val data: Inclusive = 1 to 10
    val intContent3 = NpyFile[Int]
    data.foreach(intContent3.addToBuffer)
    val bytes = intContent3.getBytes
    val intChannel3 = FileChannel.open(Paths.get("/tmp/inttest3.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel3.write(ByteBuffer.wrap(bytes))
    intChannel3.close()

/*    val data2: Inclusive = 11 to 20
    val intContent4 = NpyFile[Int]
    data2.foreach(intContent4.addToBuffer)
    val bytes2 = intContent4.getBytes
    val intChannel4 = FileChannel.open(Paths.get("/tmp/inttest4.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel4.write(ByteBuffer.wrap(bytes2))
    intChannel4.close()*/

  }
}