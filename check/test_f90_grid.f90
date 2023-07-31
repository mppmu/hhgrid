      logical function equal(a, b)
      implicit none
      real*8, intent(in) :: a, b
      if ( abs(a-b) > 1.0D-14 ) then
        equal = .false.
      else
        equal = .true.
      endif
      end function

      program testgrid
      use, intrinsic :: iso_c_binding
      implicit none
      real(c_double) :: s, t, result, expected
      type(c_ptr) :: grid
      integer :: i

      interface
        subroutine python_initialize() bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
        end subroutine python_initialize

        subroutine python_decref(grid) bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
          type(c_ptr), intent(in) :: grid
        end subroutine python_decref

        subroutine python_finalize() bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
        end subroutine python_finalize

        subroutine python_printinfo() bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
        end subroutine python_printinfo

        type(c_ptr) function grid_initialize(grid_name) bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
          character(kind=c_char) :: grid_name(*)
        end function grid_initialize

        real(c_double) function grid_virt(grid, s, t) bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
          type(c_ptr), intent(in), value :: grid
          real(c_double), intent(in), value :: s, t
        end function grid_virt
      end interface

      call python_initialize
      call python_printinfo
      grid = grid_initialize(C_CHAR_"Virt_EFT.grid"//C_NULL_CHAR)

      ! Test point 1
      s = 250000.D0
      t = -50000.D0
      expected = 3.3396514405159327719754824848052976449253D-04
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 2
      s = 66887.6D0
      t = -9407.43D0
      expected = 6.3218124421828790770609739560481621367671D-07
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 3
      s = 83055.9D0
      t =  -11438.D0
      expected = 5.7789456366871273221199473146825198455190D-06
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 4
      s = 414983.D0
      t = -59786.9D0
      expected =  1.0547862091969560302540109830715664429590D-03
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 5
      s = 2.56513D6
      t = -482321.D0
      expected = 3.7705526298104530269483802840113639831543D-02
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Evaluate point repeatedly (can detect synchronization issues with the grid)
      do i = 1, 10000
         if(.not. test_point(grid,s,t,expected)) then
            call EXIT(1)
         endif
      end do

      ! Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
      s = 275621.46328957588411867618560791015625D0;
      t = -243368.2897650574450381100177764892578125D0;
      expected = 4.2181616950034679733999576356495708751027D-04;
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
      s = 959959.10608519916422665119171142578125D0;
      t = -928446.149935275432653725147247314453125D0;
      expected = 6.0828310997215750965949609962990507483482D-03;
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      call python_decref(grid)
      call python_finalize

      write(*,*) ' '
      write(*,*) 'TESTS PASSED'
      write(*,*) ' '

      contains

        logical function test_point(grid,s,t,expected)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), intent(in), value :: grid
        real*8, intent(in), value :: s, t, expected
        logical :: equal
        real*8 :: res

        write(*,*) 'Sending: ', s,' ', t
        res = grid_virt(grid, s, t)
        write(*,*) 'Received: ', res

        if (.not. equal(res,expected)) then
          write(*,*) 'Expected: ', expected, ', Got: ', res
          write(*,*) 'TESTS FAILED'

          ! Destruct grid, terminate python
          call python_decref(grid)
          call python_finalize

          test_point = .false.
        else
          test_point = .true.
        endif
        end function

      end program
