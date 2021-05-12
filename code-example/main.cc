#include <global.hh>
#include <algorithm>
#include <RandomUnifStream.hpp>
#include <Timing.hpp>
#include <MatrixToMem.hpp>
 #ifndef __EMMINTRIN_H;
 #define __EMMINTRIN_H;
  
 #include <xmmintrin.h>
  
 typedef double __m128d __attribute__((__vector_size__(16), __aligned__(16)));
 typedef long long __m128i __attribute__((__vector_size__(16), __aligned__(16)));
  
 typedef double __m128d_u __attribute__((__vector_size__(16), __aligned__(1)));
 typedef long long __m128i_u __attribute__((__vector_size__(16), __aligned__(1)));
  
 /* Type defines.  */
 typedef double __v2df __attribute__ ((__vector_size__ (16)));
 typedef long long __v2di __attribute__ ((__vector_size__ (16)));
 typedef short __v8hi __attribute__((__vector_size__(16)));
 typedef char __v16qi __attribute__((__vector_size__(16)));
  
 /* Unsigned types */
 typedef unsigned long long __v2du __attribute__ ((__vector_size__ (16)));
 typedef unsigned short __v8hu __attribute__((__vector_size__(16)));
 typedef unsigned char __v16qu __attribute__((__vector_size__(16)));
  
 /* We need an explicitly signed variant for char. Note that this shouldn't
  * appear in the interface though. */
 typedef signed char __v16qs __attribute__((__vector_size__(16)));
  
 /* Define the default attributes for the functions in this file. */
 #define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("sse2"), __min_vector_width__(128)))
 #define __DEFAULT_FN_ATTRS_MMX __attribute__((__always_inline__, __nodebug__, __target__("mmx,sse2"), __min_vector_width__(64)))
/////////////////////////////////////////////////////////////////////////////////
//   Usage:
//           ./program_name  .......
//
//   Description:
//                ...................
//
/////////////////////////////////////////////////////////////////////////////////

void uso(std::string pname)
{
	std::cerr << "Uso: " << pname << " --fname MATRIX_FILE" << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{

	std::string fileName;
	
	//////////////////////////////////////////
	//  Read command-line parameters easy way
	if(argc != 3){
		uso(argv[0]);
	}
	std::string mystr;
	for (size_t i=0; i < argc; i++) {
		mystr=argv[i];
		if (mystr == "--fname") {
			fileName = argv[i+1];
		}
	}
	
	Timing timer0, timer1;
	////////////////////////////////////////////////////////////////
	// Transferir la matriz del archivo fileName a memoria principal
	timer0.start();
	MatrixToMem m1(fileName);
	timer0.stop();
	
	std::cout << "Time to transfer to main memory: " << timer0.elapsed() << std::endl;
	
	timer1.start();
	std::sort(m1._matrixInMemory, m1._matrixInMemory + m1._nfil);
	timer1.stop();
	
	std::cout << "Time to sort in main memory: " << timer1.elapsed() << std::endl;
	
	////////////////////////////////////////////////////////////////
	// Mostrar los 5 primeros elementos de la matriz ordenada.
	for(size_t i=0; i< 5; i++){		
		std::cout <<  m1._matrixInMemory[i] << std::endl;
	}
static __inline__ __m128i __DEFAULT_FN_ATTRS 
_mm_loadu_si32(void const *__a)
 {
   struct __loadu_si32 {
     int __v;
   } __attribute__((__packed__, __may_alias__));
   int __u = ((const struct __loadu_si32*)__a)->__v;
   return __extension__ (__m128i)(__v4si){__u, 0, 0, 0};
 }
	
//////////////////////
	//aquiempieza mi basura
std::cout <<  m1._nfil << std::endl;
	//__m128i Registro1,Registro2;
	for (size_t i=0;i< m1._nfil;i+=8){
		//auto Registro1= _mm_loadl_epi32(&m1._matrixInMemory[i]);
		//auto Registro2= _mm_loadu_si32(&m1._matrixInMemory[i+4]);

		std::cout <<  _mm_loadu_si32(&m1._matrixInMemory[i]) << std::endl;
		//std::cout <<  Registro1 << std::endl;
	}




	return(EXIT_SUCCESS);
}


