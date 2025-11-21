import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        max_z, _ = torch.max(Z, dim=self.dim, keepdim=True)
        Z_shifted = Z - max_z

        exp_z = torch.exp(Z_shifted)
        sum_exp = torch.sum(exp_z, dim=self.dim, keepdim=True)
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        self.A = exp_z / sum_exp
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            dims = list(range(len(shape)))
            dims.remove(self.dim)
            dims.append(self.dim)

            A_2d = self.A.permute(*dims).reshape(-1, C)
            dLdA_2d = dLdA.permute(*dims).reshape(-1, C)

            self.A = A_2d
            dLdA = dLdA_2d

        inner = (dLdA * self.A).sum(dim=1, keepdim=True)
        dLdZ_2d = self.A * (dLdA - inner)
        dLdZ = dLdZ_2d.reshape(*shape)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            dLdZ = dLdZ_2d.reshape(*(shape[d] for d in dims))
        
            inv_dims = [dims.index(i) for i in range(len(dims))]
            self.A = self.A.reshape(*(shape[d] for d in dims)).permute(*inv_dims)
            dLdZ = dLdZ.permute(*inv_dims)

        return dLdZ
 

    