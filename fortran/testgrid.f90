
!
! BOF - Functions required for calling grid
!

      function grid_virt(seed,s,t)
      implicit none
      real * 8 :: grid_virt
      integer ::  seed
      real * 8 :: s, t
      character(len=500) :: res  ! Buffer for function result
      character(len=500) :: arg  ! Buffer for function argument
      character(len=16) :: pyin
      character(len=17) :: pyout
      logical :: verbose

      verbose = .true.

      ! Use input seed to determine which FIFOs to use, 
      ! e.g. seed = 0 => FIFOs: pyInputPipe-0000, pyOutputPipe-0000
      pyin = "pyInputPipe-"
      pyout = "pyOutputPipe-"
      write(pyin,'(A12,I0.4)') pyin,seed
      write(pyout,'(A13,I0.4)') pyout,seed
      if (verbose) then
         write(*,*) "Using FIFOs:"
         write(*,*) pyin
         write(*,*) pyout
         write(*,*) "Input to grid_virt:"
         write(*,*) s
         write(*,*) t
      endif

      ! Build argument to be passed to python
      write(arg,'(ES50.40E3,A,ES50.40E3)') s,',',t
      if (verbose) then
         write(*,*) "Will send the following char(len=500) to python:"
         write(*,*) arg
      endif

      ! Send input to python script
      open(1,file=pyin,position='asis',action='write')
      write(1,'(A)',advance='no') arg
      close(1)

      ! Receive result from python script
      open(2,file=pyout,position='asis',action='read')
      read(2,'(A)') res
      close(2)

      ! Parse result of python grid
      if (verbose) then
         write(*,*) "Got the following character(len=500) from python:"
         write(*,*) res
      endif
      read(res,*) grid_virt
      if (verbose) then
         write(*,*) "Output of grid_virt:"
         write(*,*) grid_virt
      endif
      end function grid_virt

      function kill_python(seed)
      implicit none
      logical :: kill_python
      integer :: seed
      character(len=16) :: pyin

      ! Use input seed to determine which FIFOs to use,
      ! e.g. seed = 0 => FIFOs: pyInputPipe-0000, pyOutputPipe-0000
      pyin = "pyInputPipe-"
      write(pyin,'(A12,I0.4)') pyin,seed
      write(*,*) "Killing Python script with FIFO:"
      write(*,*) pyin

      open(1,file=pyin,position='asis',action='write')
      write(1,'(A)',advance='no') 'exit'
      close(1)

      kill_python = .true.
      end function

!
! EOF - Functions required for calling grid
!
      function equal(a,b)
      implicit none
      logical :: equal
      real * 8 :: a, b
      if ( abs(a-b) > 1.0D-14 ) then
         equal = .false.
      else
         equal = .true.
      endif
      end function

      function test_point(seed,s,t,expected)
      implicit none
      logical :: test_point
      logical :: kill_python
      integer :: seed
      real * 8 :: s, t, expected
      real * 8 :: grid_virt
      logical :: equal
      real * 8 :: res
      logical :: success
      write(*,*) 'Sending: ', s,' ', t
      res = grid_virt(seed,s,t)
      write(*,*) 'Received: ', res
      if (.not. equal(res,expected)) then
         write(*,*) 'Expected: ', expected, ', Got: ', res
         write(*,*) 'TESTS FAILED'
         success = kill_python(seed)
         test_point = .false.
      else
         test_point = .true.
      endif
      end function

      program testgrid
      implicit none
      real * 8 :: grid_virt
      logical :: kill_python, equal, test_point
      integer :: seed
      real * 8 :: s, t, res, expected
      logical :: success
      integer :: i

      ! Must match the seed used to launch grid.py
      seed = 0

      ! Test point 1
      s = 250000.D0
      t = -50000.D0
      expected = 3.3875286683507290375755305333882461127359D-04
      if(.not. test_point(seed,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 2
      s = 66887.6D0
      t = -9407.43D0
      expected = 7.6844534557689569490555349384752759078765D-07
      if(.not. test_point(seed,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 3
      s = 83055.9D0
      t =  -11438.D0
      expected = 5.6368756265788111468190146879919666389469D-06
      if(.not. test_point(seed,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 4
      s = 414983.D0
      t = -59786.9D0
      expected =  1.0669994178547940674728344845334504498169D-03
      if(.not. test_point(seed,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 5
      s = 2.56513D6
      t = -482321.D0
      expected = 3.9170655780375776555679578905255766585469D-02
      if(.not. test_point(seed,s,t,expected)) then
         call EXIT(1)
      endif

      ! Evaluate point repeatedly (can detect synchronization issues with the grid)
      do i = 1, 10000
         if(.not. test_point(seed,s,t,expected)) then
            call EXIT(1)
         endif
      end do

      ! Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
      s = 275621.46328957588411867618560791015625D0;
      t = -243368.2897650574450381100177764892578125D0;
      expected = 4.3499409718481744750728790194216344389133D-04;
      if(.not. test_point(seed,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
      s = 959959.10608519916422665119171142578125D0;
      t = -928446.149935275432653725147247314453125D0;
      expected = 5.9433502098979523395327895229911518981680D-03;
      if(.not. test_point(seed,s,t,expected)) then
         call EXIT(1)
      endif

      ! Tell python program to exit
      success = kill_python(seed)

      write(*,*) ' '
      write(*,*) 'TESTS PASSED'
      write(*,*) ' '

      end program
